---
date: 2026-06-06
tags:
  - 双机
  - 部署
  - Qwen3
  - vllm-serve
---

# Qwen3-235B-A22B 双机 vllm serve 手动启动命令

## 前置条件

- 双机 32 卡可用（使用前运行 `npu-smi info` 确认）
- vllm-ascend 代码已切到对应分支，`pip install -e .` 已装好
- vllm 代码在 `releases/v0.20.2` 分支，`pip install -e .` 已装好

---

## 启动命令

### Node 0（ master 90.90.97.40 ）

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE="1"
export OMP_PROC_BIND="false"
export OMP_NUM_THREADS="1"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_BUFFSIZE="1024"
export SERVER_PORT="8080"
export NUMEXPR_MAX_THREADS="128"
export HCCL_IF_IP="90.90.97.40"
export HCCL_SOCKET_IFNAME="enp194s0f0"
export GLOO_SOCKET_IFNAME="enp194s0f0"
export TP_SOCKET_IFNAME="enp194s0f0"
export LOCAL_IP="90.90.97.40"
export NIC_NAME="enp194s0f0"
export MASTER_IP="90.90.97.40"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

vllm serve /home/data/Qwen3-235B-A22B \
  --host 0.0.0.0 --port 8080 \
  --safetensors-load-strategy prefetch \
  --data-parallel-size 4 --data-parallel-size-local 2 \
  --data-parallel-address 90.90.97.40 --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 --seed 1024 \
  --enable-expert-parallel \
  --max-num-seqs 32 --max-model-len 8192 --max-num-batched-tokens 8192 \
  --trust-remote-code --no-enable-prefix-caching --gpu-memory-utilization 0.9
```

### Node 1（ worker 90.90.97.37 ）

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE="1"
export OMP_PROC_BIND="false"
export OMP_NUM_THREADS="1"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_BUFFSIZE="1024"
export SERVER_PORT="8080"
export NUMEXPR_MAX_THREADS="128"
export HCCL_IF_IP="90.90.97.37"
export HCCL_SOCKET_IFNAME="enp194s0f0"
export GLOO_SOCKET_IFNAME="enp194s0f0"
export TP_SOCKET_IFNAME="enp194s0f0"
export LOCAL_IP="90.90.97.37"
export NIC_NAME="enp194s0f0"
export MASTER_IP="90.90.97.40"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

vllm serve /home/data/Qwen3-235B-A22B \
  --headless \
  --safetensors-load-strategy prefetch \
  --data-parallel-size 4 --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 90.90.97.40 --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 --seed 1024 \
  --enable-expert-parallel \
  --max-num-seqs 32 --max-model-len 8192 --max-num-batched-tokens 8192 \
  --trust-remote-code --no-enable-prefix-caching --gpu-memory-utilization 0.9
```

---

## 启动顺序

1. 两台机器**同时**执行各自命令
2. 看到 `Waiting for API servers to complete` 表示服务就绪
3. 在 Node 0 上跑 benchmark 测试

## benchmark 命令（Node 0 上跑）

服务器就绪后（看到 `Waiting for API servers to complete`），在 Node 0 上通过 ais_bench CLI 运行 GSM8K accuracy 测试：

### 配置准备

ais_bench 需要两个自定义配置文件，由测试框架自动生成。手动生成命令：

```bash
cd vllm-workspace_A00275/vllm-ascend
export BENCHMARK_HOME="/tmp/aisbench_work"
mkdir -p $BENCHMARK_HOME/ais_bench/benchmark/configs/models/vllm_api
mkdir -p $BENCHMARK_HOME/ais_bench/benchmark/configs/datasets/gsm8k
mkdir -p $BENCHMARK_HOME/ais_bench/datasets
```

### Accuracy 测试（单跑）

```bash
export BENCHMARK_HOME="/tmp/aisbench_work"

# 1. 准备自定义请求配置文件
cat > $BENCHMARK_HOME/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat_custom.py << 'EOF'
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model="/home/data/Qwen3-235B-A22B",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="90.90.97.40",
        host_port=8080,
        url="",
        max_out_len=7680,
        batch_size=512,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.6,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
EOF

# 2. 参考数据集配置文件
cp /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/benchmark/configs/datasets/gsm8k/gsm8k_gen_0_shot_cot_chat_prompt.py \
   $BENCHMARK_HOME/ais_bench/benchmark/configs/datasets/gsm8k/

# 3. 拷贝 summarizer 配置
mkdir -p $BENCHMARK_HOME/ais_bench/benchmark/configs/summarizers
cp /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/benchmark/configs/summarizers/example.py \
   $BENCHMARK_HOME/ais_bench/benchmark/configs/summarizers/

# 4. 跑 accuracy benchmark
cd vllm-workspace_A00275/vllm-ascend && \
BENCHMARK_HOME="/tmp/aisbench_work" \
ais_bench --models vllm_api_general_chat_custom \
          --datasets gsm8k_gen_0_shot_cot_chat_prompt \
          --debug 2>&1 | tee /tmp/benchmark_acc.log
```

### 完整测试（perf + acc 顺序跑）

参照 CI 流程，先跑 perf 再跑 acc：

```bash
# Perf 配置
cat > $BENCHMARK_HOME/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_custom.py << 'EOF'
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="",
        model="/home/data/Qwen3-235B-A22B",
        stream=True,
        request_rate=11.2,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="90.90.97.40",
        host_port=8080,
        url="",
        max_out_len=1500,
        batch_size=700,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        ),
    )
]
EOF

# Perf 自定义数据集配置
cp /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/benchmark/configs/datasets/gsm8k/gsm8k_gen_0_shot_cot_str_perf.py \
   $BENCHMARK_HOME/ais_bench/benchmark/configs/datasets/gsm8k/

# 跑 perf
BENCHMARK_HOME="/tmp/aisbench_work" \
ais_bench --models vllm_api_stream_chat_custom \
          --datasets gsm8k_gen_0_shot_cot_str_perf_custom \
          --mode perf --num-prompts 2800 \
          --debug 2>&1 | tee /tmp/benchmark_perf.log

# 跑 acc（同上）
BENCHMARK_HOME="/tmp/aisbench_work" \
ais_bench --models vllm_api_general_chat_custom \
          --datasets gsm8k_gen_0_shot_cot_chat_prompt \
          --debug 2>&1 | tee /tmp/benchmark_acc.log
```

## 验证网卡名

如果 `enp194s0f0` 不对，在每台机器上运行：

```bash
python3 -c "import psutil,socket; [print(i) for i,a in psutil.net_if_addrs().items() for x in a if x.family==socket.AF_INET and x.address=='$(hostname -I | awk "{print \$1}")' ]"
```
