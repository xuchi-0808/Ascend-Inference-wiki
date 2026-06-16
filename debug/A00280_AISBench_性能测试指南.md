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
