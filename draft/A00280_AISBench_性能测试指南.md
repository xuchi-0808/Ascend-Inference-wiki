# AISBench 性能测试指南（A00280）

## 1 安装 AISBench

从源码安装：

```bash
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```

安装额外依赖：

```bash
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

验证安装：

```bash
ais_bench -h
```

## 2 启动 vLLM 推理服务

按文档 Chapter 5 拉起 vLLM serve 后，确保服务可访问。

## 3 配置文件

### 3.1 模型配置文件

需要修改 `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="your_model_path",         # 模型权重路径（如果不想改可以留空）
        model="qwen3",                  # vLLM serve 的 --served-model-name 值
        request_rate=0,                 # 0 = 一次性发送所有请求（满并发）
        retry=2,
        host_ip="localhost",            # vLLM 服务的 IP
        host_port=8000,                 # vLLM 服务的端口
        max_out_len=2048,               # 输出最大 token 数
        batch_size=32,                  # 最大并发数
        generation_kwargs=dict(
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            ignore_eos=True,            # 忽略结束符，确保输出达到 max_out_len
        )
    )
]
```

关键参数说明：

- `request_rate=0`：满并发压测，不控制请求间隔
- `batch_size`：最大并发数，根据测试场景调整（1~100）
- `max_out_len`：输出 token 数量，和 `--max-model-len` 的关系是 `max_out_len + 输入长度 < max_model_len`
- `ignore_eos=True`：确保模型一直输出到 `max_out_len`，否则输出长度不固定，TPOT 数据不可比

### 3.2 合成数据集配置

如果使用合成数据集（推荐，可自由控制输入输出长度），修改 `benchmark/ais_bench/datasets/synthetic/synthetic_config.py`：

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 200,                # 总请求数
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 200, "MaxValue": 2048}  # 输入长度范围
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 128, "MaxValue": 2048}   # 输出长度范围
        }
    }
}
```

常用输入输出分布：

| 场景 | MinInput | MaxInput | MinOutput | MaxOutput |
|------|----------|----------|-----------|-----------|
| 短序列 | 200 | 2048 | 128 | 2048 |
| 长序列 | 32768 | 65536 | 512 | 1024 |

## 4 执行性能测试

### 4.1 使用合成数据集

```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf
```

### 4.2 使用真实数据集

```bash
# 使用 gsm8k 数据集
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_4_shot_cot_str --mode perf

# 限制请求数（只测前 N 条）
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_4_shot_cot_str --mode perf --num-prompts 50
```

### 4.3 Debug 模式

第一次跑建议加 `--debug` 打印详细日志，方便排查问题：

```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --debug
```

### 4.4 固定并发压测（Pressure 模式）

```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --pressure
```

Pressure 模式会在指定时间内以递增并发发送请求，更接近真实负载。可以通过调整 `benchmark/ais_bench/benchmark/global_consts.py` 中的 `PRESSURE_TIME`（默认 60s）和 `CONNECTION_ADD_RATE` 控制。

## 5 结果解读

运行完成后，终端会直接打印性能结果表格，包含以下核心指标：

| 指标 | 含义 | 关注场景 |
|------|------|---------|
| **TTFT** | 首 token 延迟 (ms) | 低时延场景关注 |
| **TPOT** | 每 token 延迟 (ms) | 低时延场景关注 |
| **ITL** | token 间延迟 (ms) | 同 TPOT |
| **OutputTokenThroughput** | 输出吞吐 (token/s) | 高吞吐场景关注 |
| **E2EL** | 端到端延迟 (ms) | 综合参考 |
| **Concurrency** | 实际达到的并发数 | 验证是否达到预期并发 |

结果文件默认保存在 `outputs/default/<timestamp>/performances/vllm-api-stream-chat/` 下：

- `*.csv`：单次请求逐条数据
- `*.json`：汇总数据
- `*_plot.html`：请求并发可视化图表（浏览器打开）

## 6 常见问题

### Q: 请求全部失败 / HTTP 报错

检查模型配置文件中的 `host_ip` 和 `host_port` 是否与 vLLM 服务一致。

### Q: 实际并发数达不到设定值

检查服务端显存是否足够。降低 `batch_size` 或 `max_out_len`。

### Q: 输出 token 数达不到 max_out_len

确认模型配置文件中 `generation_kwargs` 内已设置 `ignore_eos=True`。

### Q: 找不到数据集

确认数据集已下载并放在 `ais_bench/datasets/` 下。合成数据集不需要下载。

## 7 复现记录：A3 测试方案 Config B（低时延 TP4）

### 7.1 测试环境

| 项目 | 内容 |
|------|------|
| 服务器 | Atlas 800I A3（S3），8 NPU × 2 chip = 16 chips |
| 镜像 | `quay.io/ascend/vllm-ascend:nightly-main-a3`（commit `89737c862d42`） |
| vLLM 版本 | 0.21.0 |
| vLLM-Ascend 版本 | 0.19.1rc2.dev544 |
| 模型 | Qwen3-30B-A3B-W8A8（`/home/weights/Eco-Tech/Qwen3-30B-A3B-w8a8`） |
| Eagle3 权重 | `/home/weights/AngelSlim/Qwen3-a3B_eagle3` |
| 测试日期 | 2026-06-17 |

### 7.2 服务端配置（Config B）

对应测试方案 Section "配置 B：低时延 (TP4, 2 卡)"，完整启动脚本：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /home/weights/Eco-Tech/Qwen3-30B-A3B-w8a8 \
    --served-model-name qwen3 \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --port 8000 \
    --distributed_executor_backend mp \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --speculative-config '{"method": "eagle3","model": "/home/weights/AngelSlim/Qwen3-a3B_eagle3", "num_speculative_tokens": 3}'
```

### 7.3 AISBench 测试配置

**模型配置**（`vllm_api_stream_chat.py`）：

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="/home/weights/Eco-Tech/Qwen3-30B-A3B-w8a8",
        model="qwen3",
        stream=True,
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8000,
        max_out_len=2048,
        batch_size=1,               # 并发=1，对应测试方案第一行
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=True,
        ),
    )
]
```

**合成数据集**（`synthetic_config.py`）：

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 4,              # 对应测试方案并发=1 时的 4 条请求
    "StringConfig": {
        "Input":  {"Method": "uniform", "Params": {"MinValue": 2048, "MaxValue": 2048}},
        "Output": {"Method": "uniform", "Params": {"MinValue": 2048, "MaxValue": 2048}}
    }
}
```

**执行命令**：

```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --num-prompts 4
```

### 7.4 结果对比

测试方案 Section 2.1（并发=1，输入 2048 / 输出 2048）参考值与本次实测对比：

| 指标 | 参考值 (v0.13.0) | 本次实测 (v0.21.0) | 变化 |
|------|-----------------|-------------------|------|
| 总请求数 | 4 | 4 | — |
| 并发 | 1 | 1.0 | — |
| 总时长 (s) | 60.6 | **38.4** | ↓ 37% |
| Output 吞吐 (tps) | 135.1 | **213.2** | ↑ 58% |
| 总吞吐 (tps) | 270.8 | **427.3** | ↑ 58% |
| 有效 TPOT (ms/token) | 7.2 | **4.7** | ↓ 35% |
| 估算 TTFT (ms) | 359.6 | **~193** | ↓ 46% |
| Prefill 吞吐 (tps) | — | 10641.1 | — |

### 7.5 分析

- **复现成功**：Config B 低时延场景在 TP4 + EP + FlashComm + Eagle3 配置下跑通，4 请求全部成功
- **性能优于参考值**：各项指标显著提升，主要原因：
  - vLLM-Ascend 版本从 v0.13.0 升级到 v0.21.0（约 8 个版本迭代），kernel 优化和 Eagle3 实现改进带来显著提速
  - `--async-scheduling` 从默认关闭变为显式启用，调度效率提升
  - ACL graph memory profiling（v0.21.0 新增）优化了显存利用率
- **TPOT 从 7.2ms 降到 4.7ms**：Eagle3 投机解码 + async scheduling 的组合效果，每 token 生成速度提升 35%
- **TTFT 从 360ms 降到 ~193ms**：prefill 阶段优化（FlashComm + 编译优化）

### 7.6 注意事项

- 参考方案使用 `ASCEND_RT_VISIBLE_DEVICES=12,13,14,15`，本次实测使用 `0,1,2,3`（A3 机器 NPU 编号不同，性能无差异）
- 参考方案端口为 1999，本次使用 8000（不影响性能）
- 本次测试 `ignore_eos=True`，输出严格 2048 tokens；参考方案的输出长度策略未注明，可能导致吞吐计算口径略有差异
