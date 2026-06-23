# A00282 — Nightly 环境复现指南

> 2026-06-23 创建。适用于 Qwen3-235B-A22B 多机多卡 GSM8K accuracy 测试的完整环境复现。
> 本文版本配套信息均从 GitHub Actions 原始日志逐行提取（`20260509_last_success.log` / `20260510_first_fail.log`），**可用于精确复现**。

## 版本配套表（日志实证）

### 核心代码

| 组件 | 5/9 成功版 | 5/10 失败版 |
|------|-----------|-----------|
| **vllm** | `d886c26d4`（`0.1.dev1+gd886c26d4.empty`）<br>作者: Lxx `<lyfqlx3@gmail.com>`<br>日期: 2026-04-19<br>信息: `[Doc] Fix typos in token_embed pooling documentation (#40266)` | `4d51588e2`（`0.1.dev1+g4d51588e2.empty`）<br>作者: Yifan Qiao `<yifanqiao@inferact.ai>`<br>日期: 2026-04-26<br>信息: `[Feat] DeepSeek V4 Rebased (#40860)` |
| **vllm-ascend** | `68ff5263`（`0.19.1rc2.dev28+g68ff52636`）<br>作者: guanguan0308<br>日期: 2026-05-08<br>信息: `[BugFix] xmask feature for dispatch_ffn_combine operator (#8789)` | `ca4065f2`（`0.19.1rc2.dev42+gca4065f2e`）<br>日期: 2026-05-09 |

### Ascend 底层栈（5/9 ↔ 5/10 完全一致）

| 层级 | 组件 | 版本 |
|------|------|------|
| **CANN Toolkit** | package | `Ascend-cann-toolkit` |
| | 版本号 | `8.5.1` |
| | innerversion | `V100R001C25SPC002B220` |
| | 兼容最低 | `V100R001C15` |
| | 安装路径 | `/usr/local/Ascend/cann-8.5.1` |
| **NPU 驱动** | npu-smi | `25.5.2` |
| | NPU 型号 | `910B3` |
| **Triton** | triton-ascend | `3.2.0`<br>来源: `https://gitcode.com/Ascend/triton-ascend/`<br>位置: `/usr/local/python3.11.14/lib/python3.11/site-packages` |
| **bishengir** | bishengir-compile | 由 CANN 8.5.1 安装<br>路径: `/usr/local/Ascend/cann-8.5.1/tools/bishengir/bin/bishengir-compile` |

### Python 运行环境（5/9 ↔ 5/10 完全一致）

| 组件 | 版本 | 备注 |
|------|------|------|
| **Python** | `3.11.14` | 路径: `/usr/local/python3.11.14/bin/python3` |
| **pytest** | `8.4.2` | 插件: asyncio-1.3.0, cov-7.1.0, mock-3.15.1, anyio-4.13.0 |
| **pluggy** | `1.6.0` | — |
| **ais_bench** | `3.1.20260429`（对应 tag `v3.1-20260429-master`）<br>位置: `/vllm-workspace/vllm-ascend/benchmark`（可编辑安装） | — |

### 系统与编译器（5/9 ↔ 5/10 完全一致）

| 项 | 值 |
|----|-----|
| **OS** | Ubuntu 22.04（aarch64） |
| **clang** | `15.0.7`（`/usr/bin/clang-15`） |
| **GCC** | `11`（`/usr/bin/../lib/gcc/aarch64-linux-gnu/11`） |
| **架构** | `aarch64` |
| **vllm install 方式** | `VLLM_TARGET_DEVICE="empty"`, editable install (`/vllm-workspace/vllm/`) |
| **vllm-ascend install 方式** | pip install `-e`, editable (`/vllm-workspace/vllm-ascend/`) |

### vllm serve 关键编译配置（5/9 ↔ 5/10 完全一致）

| 配置项 | 值 |
|--------|-----|
| **compilation_mode** | `VLLM_COMPILE: 3`（`VLLM_COMPILE`） |
| **cudagraph_mode** | `PIECEWISE: 1` |
| **num_of_warmups** | `1` |
| **capture_sizes** | `[1, 56, 128]` |
| **max_capture_size** | `128` |
| **compile_backend** | `vllm_ascend.compilation.compiler_interface.AscendCompiler` |
| **inductor: combo_kernels** | `True` |
| **inductor: size_asserts** | `False`（所有 assert 关） |
| **pass_config: fuse_norm_quant** | `True` |
| **pass_config: fuse_act_quant** | `True` |
| **pass_config: fuse_attn_quant** | `False` |
| **pass_config: enable_sp** | `False` |
| **pass_config: fuse_gemm_comms** | `False` |
| **pass_config: fuse_allreduce_rms** | `False` |
| **moe_backend** | `auto` |
| **enable_flashinfer_autotune** | `True` |
| **ir_op_priority: rms_norm** | `['native']` |

> **结论（日志实证）**：5/9 → 5/10 之间 **只有 vllm + vllm-ascend 代码变化**。CANN/驱动/triton/Python/pytest/系统/编译配置 **全部一致**。回归主体是 vllm upstream `d886c26d4 → 4d51588e2`（199 commits），vllm-ascend 的 `7fd2cede` 只是随之做的适配性改动。

### 完整 diff 命令

```bash
# vllm：从成功版切到失败版
git log --oneline d886c26d4..4d51588e2

# vllm-ascend：从成功版切到失败版
git log --oneline 68ff5263..ca4065f2
```

## 路径 A：自建镜像（从 CANN base 构建，需可访问 quay.io/SWR）

模拟 CI 的 `schedule_image_build_and_push.yaml` → `_nightly_image_build.yaml` 双层构建流程，用单个 Dockerfile 从 CANN base 直出。

### Dockerfile

以下 Dockerfile 等价于 CI 的 `quay.io/ascend/vllm-ascend:nightly-main` + `swr.cn-southwest-2:nightly-ci-main-a2` 的组合效果。

> **坑：quay.io 在国内网络可能无法访问**。如果 `FROM quay.io/ascend/cann:...` pull 超时，
> 改用内网 SWR 的 CANN 镜像（见下方备选 FROM 行）。

```dockerfile
# ===== Stage 1: Base image =====
# 注意：CANN 8.5.1 + Python 3.11 是 5/9 nightly 的 exact match
#
# 备选 FROM（按优先级）：
# 1. 内网 SWR（推荐，CI 也在用）:
#    FROM swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-910b-ubuntu22.04-py3.12
# 2. quay.io（需外网访问）:
#    FROM quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11
FROM quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11

ARG PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
ARG VLLM_COMMIT=d886c26d4
ARG VLLM_ASCEND_COMMIT=68ff5263
ARG MOONCAKE_TAG="v0.3.9"
ARG SOC_VERSION="ascend910b1"

# --- System deps ---
RUN apt-get update -y && \
    apt-get install -y git vim wget net-tools gcc g++ cmake numactl libnuma-dev \
                       libjemalloc2 clang-15 pciutils && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20 && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# --- Mooncake (optional: for multi-node transfer) ---
RUN git clone --depth 1 --branch ${MOONCAKE_TAG} \
        https://github.com/kvcache-ai/Mooncake /vllm-workspace/Mooncake && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    mkdir -p /vllm-workspace/Mooncake/build && \
    cd /vllm-workspace/Mooncake/build && \
    cmake .. -DUSE_ASCEND_DIRECT=ON && \
    make -j$(nproc) && make install && \
    rm -rf /vllm-workspace/Mooncake/build

# --- modelscope + ray ---
RUN pip config set global.index-url ${PIP_INDEX_URL} && \
    python3 -m pip install modelscope 'ray>=2.47.1,<=2.48.0' 'protobuf>3.20.0'

# --- Install vllm at target commit ---
RUN git init /vllm-workspace/vllm && \
    git -C /vllm-workspace/vllm fetch --depth 1 \
        https://github.com/vllm-project/vllm.git ${VLLM_COMMIT} && \
    git -C /vllm-workspace/vllm checkout FETCH_HEAD && \
    VLLM_TARGET_DEVICE="empty" python3 -m pip install -e /vllm-workspace/vllm/[audio] \
        --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip uninstall -y triton && \
    python3 -m pip cache purge

# --- Install vllm-ascend at target commit ---
ENV DEBIAN_FRONTEND=noninteractive
ENV SOC_VERSION=${SOC_VERSION} \
    TASK_QUEUE_ENABLE=1 \
    OMP_NUM_THREADS=1

RUN git init /vllm-workspace/vllm-ascend && \
    git -C /vllm-workspace/vllm-ascend fetch --depth 1 \
        https://github.com/vllm-project/vllm-ascend.git ${VLLM_ASCEND_COMMIT} && \
    git -C /vllm-workspace/vllm-ascend checkout FETCH_HEAD && \
    export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi" && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    python3 -m pip install -e /vllm-workspace/vllm-ascend/ \
        --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip uninstall -y triton triton-ascend && \
    python3 -m pip install triton-ascend==3.2.0 \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi && \
    python3 -m pip cache purge

# ===== Stage 2: Nightly CI layer =====
# 等价于 Dockerfile.nightly.a2 做的事

# --- requirements-dev.txt ---
WORKDIR /vllm-workspace/vllm-ascend
RUN export PIP_EXTRA_INDEX_URL="https://repo.huaweicloud.com/ascend/repos/pypi" && \
    python3 -m pip install -r requirements-dev.txt && \
    python3 -m pip cache purge

# --- AISBench benchmark ---
RUN git clone -b v3.1-20260429-master --depth 1 \
        https://github.com/AISBench/benchmark.git \
        /vllm-workspace/vllm-ascend/benchmark && \
    cd /vllm-workspace/vllm-ascend/benchmark && \
    pip install -e . -r requirements/api.txt -r requirements/extra.txt && \
    python3 -m pip cache purge

# --- Environment setup ---
RUN echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:\$LD_PRELOAD" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib" >> ~/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
```

### 构建命令

**Dockerfile 不含 `COPY`，不要在大目录下执行！** build context 会被全部打包到 Docker daemon，浪费数分钟传输几十 GB 的空数据。

推荐两种方式避免：

**方式 1：空目录构建（推荐）**

```bash
mkdir -p /tmp/docker-build && cd /tmp/docker-build
# 把 Dockerfile.nightly.repro 移到或复制到此目录
cp /path/to/Dockerfile.nightly.repro .

IMAGE_TAG="vllm-nightly-repro:5-9-success"
docker build \
  --network host \
  --build-arg VLLM_COMMIT=d886c26d4 \
  --build-arg VLLM_ASCEND_COMMIT=68ff5263 \
  -t "$IMAGE_TAG" \
  -f Dockerfile.nightly.repro .
```

**方式 2：stdin 输入（不需要移动文件）**

```bash
IMAGE_TAG="vllm-nightly-repro:5-9-success"
docker build \
  --network host \
  --build-arg VLLM_COMMIT=d886c26d4 \
  --build-arg VLLM_ASCEND_COMMIT=68ff5263 \
  -t "$IMAGE_TAG" \
  - < Dockerfile.nightly.repro
```

> 构建机是 aarch64（ARM），无需指定 `--platform`。若在 x86 构建 aarch64 镜像，加 `--platform linux/arm64`。

### 构建耗时预估

- CANN base 层 pull：~5-10min（镜像 ~8GB）
- Mooncake 编译：~3min
- vllm pip install（empty device）：~2min
- vllm-ascend pip install（含 csrc 编译）：~5-8min
- 总耗时：~20-30min

---

## 路径 B：已有镜像组件升级

适用于已有 vllm-ascend 镜像（如 `quay.io/ascend/vllm-ascend:nightly-main` 或任何自制镜像），在容器内替换组件版本。

### 容器内升级脚本

```bash
#!/bin/bash
# reproduce_env_setup.sh — 在已有镜像中复现 nightly 环境
set -euo pipefail

# ===== Config =====
VLLM_COMMIT="${1:-d886c26d4}"
VLLM_ASCEND_COMMIT="${2:-68ff5263}"
TRITON_ASCEND_VER="${3:-3.2.0}"
AISBENCH_TAG="${4:-v3.1-20260429-master}"
WORKSPACE="${WORKSPACE:-/vllm-workspace}"

# ===== Step 1: Replace vllm =====
echo "[1/5] Installing vllm @ ${VLLM_COMMIT}..."
cd "$WORKSPACE"
if [ -d vllm ]; then
    rm -rf vllm  # 删掉旧的可编辑安装
fi
git init vllm
git -C vllm fetch --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_COMMIT"
git -C vllm checkout FETCH_HEAD
VLLM_TARGET_DEVICE="empty" pip install -e vllm/[audio] \
    --extra-index https://download.pytorch.org/whl/cpu/
pip uninstall -y triton 2>/dev/null || true
pip cache purge

# ===== Step 2: Replace vllm-ascend =====
echo "[2/5] Installing vllm-ascend @ ${VLLM_ASCEND_COMMIT}..."
cd "$WORKSPACE"
if [ -d vllm-ascend ]; then
    cp -r vllm-ascend/benchmark /tmp/aisbench-backup 2>/dev/null || true
    rm -rf vllm-ascend
fi
git init vllm-ascend
git -C vllm-ascend fetch --depth 1 https://github.com/vllm-project/vllm-ascend.git "$VLLM_ASCEND_COMMIT"
git -C vllm-ascend checkout FETCH_HEAD

export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -e vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/
pip cache purge

# ===== Step 3: Fix triton-ascend version =====
echo "[3/5] Installing triton-ascend==${TRITON_ASCEND_VER}..."
pip uninstall -y triton triton-ascend 2>/dev/null || true
pip install "triton-ascend==${TRITON_ASCEND_VER}" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

# ===== Step 4: Install requirements-dev.txt =====
echo "[4/5] Installing requirements-dev.txt..."
cd "$WORKSPACE/vllm-ascend"
pip install -r requirements-dev.txt
pip cache purge

# ===== Step 5: Install AISBench =====
echo "[5/5] Installing AISBench (${AISBENCH_TAG})..."
if [ -d "$WORKSPACE/vllm-ascend/benchmark" ]; then
    cp -r /tmp/aisbench-backup "$WORKSPACE/vllm-ascend/benchmark" 2>/dev/null || true
fi
cd "$WORKSPACE/vllm-ascend/benchmark"
pip install -e . -r requirements/api.txt -r requirements/extra.txt
pip cache purge

echo ""
echo "=== Environment ready ==="
pip list 2>/dev/null | grep -E "vllm|triton|ais_bench" || true
```

### 使用方法

```bash
# 在容器或镜像内执行
bash reproduce_env_setup.sh d886c26d4 68ff5263 3.2.0 v3.1-20260429-master
```

### 验证版本

```bash
pip list | grep -E "vllm|triton|ais_bench"
# 期望输出类似：
# vllm                 0.1.dev1+gd886c26d4      /vllm-workspace/vllm
# vllm_ascend          0.19.1rc2.dev28+g68ff52636 /vllm-workspace/vllm-ascend
# triton-ascend        3.2.0
# ais_bench_benchmark  3.1.20260429             /vllm-workspace/vllm-ascend/benchmark

npu-smi info
# 期望 Version: 25.5.2

cat /usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/ascend_toolkit_install.info
# 确认 version=8.5.1
```

---

## 运行测试

### 方式一：pytest 自动化（推荐，与 CI 一致）

**环境变量设置：**

```bash
export WORKSPACE=/vllm-workspace
export CONFIG_YAML_PATH=tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-A22B-A2.yaml
export FAIL_TAG=FAIL_TAG
export IS_PR_TEST=false
export BENCHMARK_JOB_NAME=my-repro
export VLLM_CI_RUNNER=a2

# 多机跨机 NCCL 配置（根据实际网络调整）
export HCCL_IF_IP=<本机IP>
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

# modelscope 离线模式（模型须提前下载）
export VLLM_USE_MODELSCOPE=True
export HF_HUB_OFFLINE=1

# 运行 pytest
cd /vllm-workspace/vllm-ascend
pytest -sv tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py
```

**pytest 内部流程：**

1. 加载 `Qwen3-235B-A22B-A2.yaml` 配置
2. 解析环境变量和 `vllm serve` 命令参数
3. 启动 `vllm serve Qwen/Qwen3-235B-A22B --data-parallel-size 2 --tensor-parallel-size 8 ...`
4. 等待 server ready（超时 2800s）
5. 依次运行 perf + acc benchmark（通过 ais_bench）

**关键 GPU/NPU 参数（取自 config YAML）：**

```bash
# Leader 节点
vllm serve "Qwen/Qwen3-235B-A22B" \
  --host 0.0.0.0 --port 8080 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address $LOCAL_IP \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --enable-expert-parallel \
  --max-num-seqs 128 \
  --max-model-len 40960 \
  --max-num-batched-tokens 2048 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9

# Worker 节点（headless）
vllm serve "Qwen/Qwen3-235B-A22B" \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $MASTER_IP \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --max-num-seqs 128 \
  --max-model-len 40960 \
  --max-num-batched-tokens 2048 \
  --enable-expert-parallel \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

### 方式二：手动 vllm serve + ais_bench（灵活调试）

**Step 1: 拉起 serve（两个节点）**

Leader 节点：

```bash
export VLLM_USE_MODELSCOPE=True
export OMP_PROC_BIND=False
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1

vllm serve "Qwen/Qwen3-235B-A22B" \
  --host 0.0.0.0 --port 8080 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address <LEADER_IP> \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --enable-expert-parallel \
  --max-num-seqs 128 \
  --max-model-len 40960 \
  --max-num-batched-tokens 2048 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

Worker 节点：

```bash
# 同样的环境变量
vllm serve "Qwen/Qwen3-235B-A22B" \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address <LEADER_IP> \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --max-num-seqs 128 \
  --max-model-len 40960 \
  --max-num-batched-tokens 2048 \
  --enable-expert-parallel \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

**Step 2: 运行 accuracy 测试**

```bash
export BENCHMARK_HOME=/vllm-workspace/vllm-ascend/benchmark

ais_bench \
  --model Qwen/Qwen3-235B-A22B \
  --port 8080 \
  --dataset vllm-ascend/gsm8k-lite \
  --request-conf vllm_api_general_chat \
  --dataset-conf gsm8k/gsm8k_gen_0_shot_cot_chat_prompt \
  --max-out-len 7680 \
  --batch-size 256 \
  --case-type accuracy
```

**Step 3: 运行 perf 测试（可选）**

```bash
ais_bench \
  --model Qwen/Qwen3-235B-A22B \
  --port 8080 \
  --dataset vllm-ascend/GSM8K-in3500-bs2800 \
  --request-conf vllm_api_stream_chat \
  --dataset-conf gsm8k/gsm8k_gen_0_shot_cot_str_perf \
  --num-prompts 2800 \
  --max-out-len 1500 \
  --batch-size 256 \
  --request-rate 4.8 \
  --case-type performance
```

---

## 常见问题

### Q: quay.io 拉不下来（DNS 超时 / 连接拒绝）？

内网环境通常无法直接访问 quay.io。解决方案：

1. **改用内网 SWR 的 CANN 镜像**（推荐，CI 也用此源）：

   ```dockerfile
   FROM swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-910b-ubuntu22.04-py3.12
   ```

   注意这是 CANN 9.0.0 + Python 3.12（5/9 是 8.5.1 + 3.11），但 vllm/vllm-ascend 兼容。

2. **用已有的 nightly-ci 镜像做 base，只升级组件**（即 Path B）：

   ```dockerfile
   FROM swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/vllm-ascend:nightly-ci-main-a2
   # 然后在此镜像上替换 vllm/vllm-ascend 到目标 commit
   ```

3. **先手动 pull CANN 镜像到本地，改 Dockerfile 使用本地 tag**：

   ```bash
   # 找一台能访问 quay.io 的机器 pull 下来，docker save 成 tar，scp 到内网 docker load
   docker pull quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11
   docker save quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11 | gzip > cann-8.5.1.tar.gz
   # 内网:
   docker load < cann-8.5.1.tar.gz
   # 用本地镜像名
   docker tag quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11 cann:8.5.1-local
   # Dockerfile 中改为 FROM cann:8.5.1-local
   ```

### Q: docker build 时 build context 很大（几十 GB）？

Dockerfile 不含 `COPY` 指令，不需要任何 build context。**不要在当前目录下执行 `docker build .`**。

做法见上方「构建命令」的两种方式：空目录构建或用 `- < Dockerfile` 的 stdin 模式。

### Q: CANN 版本不匹配怎么办？

当前 `quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11` 可能已下架。替代方案：

当前 `quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11` 可能已下架。替代方案：

1. **用 9.0.0 base**（当前 CI 用的版本）：把 FROM 改成 `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12`，Python 从 3.11 改 3.12。**注意**：vllm + vllm-ascend 代码在 Python 3.12 下兼容，但 triton-ascend 需确认有 3.12 wheel。
2. **联系 CI 团队**要 5/9 那天的 nightly 镜像 digest：`swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/vllm-ascend:nightly-ci-main-a2@sha256:...`，用 digest 拉取不可变引用。

### Q: 只有单机，没有双机怎么办？

把 `data-parallel-size` 改成 1（不跨机 DP），仅 TP=8 做单机精度验证：

```bash
# 去掉 --data-parallel-size / --data-parallel-address / --data-parallel-rpc-port
vllm serve "Qwen/Qwen3-235B-A22B" \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --max-num-seqs 128 \
  --max-model-len 40960 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

**注意**：单机没有 DP，不会复现 MC2 路径的跨机并发问题（本 Issue 根因）。但可用于验证 vllm + vllm-ascend 版本配对是否正常。

### Q: AISBench 数据集怎么获取？

数据集头次使用时 ais_bench 会自动下载。如需离线准备：

```bash
# GSM8K
wget https://huggingface.co/datasets/openai/gsm8k/resolve/main/gsm8k_v1.1.tar.gz
```

数据集路径由 `--dataset` 和 `--dataset-conf` 参数指定，格式为 `{dataset_name}/{config_name}`。

### Q: 需要验证组件版本的 CUJ（关键用户旅程）？

```bash
# 1. 确认 CANN
cat /usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/ascend_toolkit_install.info

# 2. 确认 NPU
npu-smi info

# 3. 确认 vllm 版本
python3 -c "import vllm; print(vllm.__version__)"

# 4. 确认 vllm-ascend 版本
python3 -c "import vllm_ascend; print(vllm_ascend.__version__)" 2>/dev/null || \
  pip show vllm-ascend | grep Version

# 5. 确认 triton-ascend
pip show triton-ascend | grep -E "Version|Location"

# 6. 确认 AISBench
pip list 2>/dev/null | grep ais_bench

# 7. 确认 modelscope 可用（模型已缓存）
python3 -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-235B-A22B', local_files_only=True))"
```

---

## 附录：CI 镜像构建链完整结构

```text
quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11              ← 华为官方 CANN 镜像
  │
  ├─ Dockerfile (vllm-ascend 项目根 Dockerfile)
  │   ├─ apt: git, clang-15, cmake, numactl, jemalloc...
  │   ├─ Mooncake 编译安装
  │   ├─ pip: modelscope, ray
  │   ├─ pip: vllm @ commit X (empty device, editable)
  │   ├─ pip: vllm-ascend @ commit Y (editable + csrc 编译)
  │   └─ pip: triton-ascend == 3.2.0
  │
  ├→ quay.io/ascend/vllm-ascend:nightly-main                    ← 上游 base 镜像
  │
  ├─ Dockerfile.nightly.a2 (ci 额外层)
  │   ├─ pip: requirements-dev.txt
  │   └─ pip: AISBench benchmark (可编辑安装)
  │
  ├→ swr.cn-southwest-2:nightly-ci-main-a2                     ← nightly CI 使用镜像
  │
  ├─ K8s LWS 部署 → 容器启动 → run.sh
  │   ├─ (IS_PR_TEST=false, 跳过 checkout)
  │   └─ pytest test_multi_node.py
  │
  └→ 加载 config YAML → vllm serve → ais_bench perf + acc
```

### 对应的工作流文件

| CI 环节 | GitHub Workflow |
|---------|----------------|
| 构建上游 base 镜像 | `.github/workflows/schedule_image_build_and_push.yaml`（调用 `_schedule_image_build.yaml`） |
| 构建 nightly-ci 镜像 | `.github/workflows/_nightly_image_build.yaml` |
| 编排 nightly 测试（构建 + 单机 + 多机） | `.github/workflows/schedule_nightly_test_a2.yaml` |
| 部署并运行 nightly 多机测试 | `.github/workflows/_e2e_nightly_multi_node.yaml` |

### 路径 A Dockerfile vs CI 对照

| Dockerfile 中的步骤 | 对应 CI 的哪个阶段 |
|---------------------|------------------|
| `FROM quay.io/ascend/cann:...` | schedule_image_build_and_push 的 base |
| 系统工具 + Mooncake | `Dockerfile`（vllm-ascend 根 Dockerfile） |
| modelscope + ray | `Dockerfile` |
| vllm @ commit | `Dockerfile`（VLLM_COMMIT build-arg） |
| vllm-ascend @ commit | `Dockerfile`（checkout 对应 ref） |
| triton-ascend | `Dockerfile` |
| requirements-dev.txt | `Dockerfile.nightly.a2` |
| AISBench benchmark | `Dockerfile.nightly.a2` |
