---
date: 2026-06-24
categories:
  - 缺陷分析
tags:
  - vllm-ascend
  - 量化
  - NetLoader
---

# vLLM-Ascend NetLoader 量化加载缺陷代码层面分析

## 背景

社区反馈：使用 vLLM-Ascend 的 NetLoader 做权重 device-to-device (d2d) 加载时，遇到一系列与量化相关的崩溃问题：

1. 不同模型的量化方式不同，主模型和 MTP (draft) 的量化也可能不同，NetLoader 如何区分？
2. DeepSeek-V4 (dsv4) 试过不行：权重 d2d 加载本身能做，但一推理就崩溃
3. 崩溃现象：**量化权重变成了三维**
4. 感觉这个功能变成了 "model specific"，没法通用

本文从代码层面分析 NetLoader 当前实现存在哪些缺陷、在什么量化 / 模型场景下会遇到什么问题。**本文只做静态代码分析，不做 NPU 实测验证**。

所有文件路径相对 `vllm-ascend/` 仓根目录，上游 vLLM 路径相对 `vllm/` 仓根目录。

---

## 1. NetLoader 架构速览

### 1.1 模块结构

| 角色 | 文件 | 关键符号 |
|---|---|---|
| 入口注册 | `vllm_ascend/model_loader/netloader/__init__.py:18` | `register_netloader()` |
| 主类 | `vllm_ascend/model_loader/netloader/netloader.py:39` | `ModelNetLoaderElastic(BaseModelLoader)`，注册名 `"netloader"` |
| d2d 入口 | `vllm_ascend/model_loader/netloader/load.py` | `elastic_load()` |
| 接收端 | `vllm_ascend/model_loader/netloader/executor/elastic_load.py` | `P2PLoad`（receiver） |
| 发送端 | 同上 | `P2PSend`（sender） |
| Socket 协调 | `vllm_ascend/model_loader/netloader/interaction/elastic.py` | `ElasticServer` / `ElasticClient` |
| HCCL PG | `vllm_ascend/model_loader/netloader/executor/netloader_pg.py` | `stateless_init_process_group()` |

### 1.2 `load_model` 主流程

入口在 `netloader.py:139-328`。关键步骤（行号对应当前 main）：

```python
# netloader.py:139
def load_model(self, vllm_config, model_config, prefix=""):
    # L155
    need_process_weights_after_loading = False

    # L169-183: source 配置无效 → 直接走 DefaultModelLoader
    if self.source is None or device_id not in [...]:
        model, need_process_weights_after_loading = self.revert_to_default(...)
    else:
        # L187-189: 备份 quant_config 和 model_config（仅备份，不清除）
        _quant_config = deepcopy(vllm_config.quant_config)
        model_config_backup = deepcopy(model_config)

        # L191-193: initialize_model —— 此时 quant_config 仍在 vllm_config 中，
        #           所以量化方法会按"未处理"状态创建 weight、weight_scale 等 param
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config, model_config, prefix)

        # L195-222: d2d 接收权重
        model = elastic_load(model=model, device_id=device_id, ...)

        # L225: 关键标记 —— 无论 sender 发的是什么，receiver 都会再跑一次 process
        need_process_weights_after_loading = True

        # L227-245: 若 elastic_load 失败，回退到 DefaultModelLoader
        if model is None:
            ...
            model, need_process_weights_after_loading = self.revert_to_default(...)

    # L247-319: 启动 ElasticServer，让本 rank 作为后续 rank 的 sender
    elastic_server = ElasticServer(driver_ip, listen_port, model, ...)
    elastic_server.start()

    # L321-322: 最后跑一次上游 process_weights_after_loading
    if need_process_weights_after_loading:
        process_weights_after_loading(model, model_config, torch.device(...))
```

### 1.3 d2d 权重传输协议（关键）

发送端 `P2PSend.send`（`executor/elastic_load.py`）：

```python
for name, param in model.named_parameters():
    if "aclnn_input_scale" in name:           # 唯一 hardcode 的跳过
        continue
    if name in int8_params:                   # HBM/DRAM 备份的 int8 weight
        sender_pg.send([int8_params[name].to(model_device)], 0, 0).wait()
    else:
        sender_pg.send([param.contiguous()], 0, 0).wait()
torch.distributed.barrier(group=sender_pg, device_ids=[model_device.index])
```

接收端 `P2PLoad.load`（`executor/elastic_load.py`）：

```python
for name, param in model.named_parameters():
    if len(param.shape) == 0:                 # 跳过标量
        continue
    receiver_pg.recv([param], 1, 0).wait()    # 直接覆写 param.data
torch.distributed.barrier(group=receiver_pg, device_ids=[model_device.index])
```

**协议特征**：

- 基于 HCCL point-to-point send/recv，无 shape / dtype / 顺序协商
- 两边各自遍历 `model.named_parameters()`，靠顺序匹配
- 没有任何 manifest 交换（参数名、shape、dtype、是否已 processed）
- 唯一的"差异化处理"是 sender 侧 hardcode 跳过名字含 `aclnn_input_scale` 的 param

### 1.4 上游 `process_weights_after_loading` 调用时机

`vllm/model_executor/model_loader/utils.py:100-127`：

```python
def process_weights_after_loading(model, model_config, target_device):
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
            release_device_memory_under_pressure(target_device)
    # 再处理 Attention / MLA / MM Encoder
    ...
```

上游 `BaseModelLoader.load_model` 在 `base_loader.py:80` 内部调用它。所以对 sender 而言：当 `ModelNetLoaderElastic.load_model` 走到 L247 启动 `ElasticServer` 时，model 已经走完了 `process_weights_after_loading` —— **sender 发出的是已 processed 的权重**。

---

## 2. 缺陷地图

### 2.1 【致命】缺陷 1：Sender 与 Receiver 的处理状态不对齐

#### 2.1.1 时序错位

| 阶段 | Sender (rank 0，已加载完) | Receiver (rank 1，正在加载) |
|---|---|---|
| initialize_model | 已完成（参数 shape 为 raw） | 已完成（参数 shape 为 raw） |
| **process_weights_after_loading** | **已执行** → shape 可能变 3D / 4D | **未执行** → shape 仍是 raw 2D |
| elastic_load (d2d) | 遍历 processed 后的 named_parameters 发送 | 遍历 raw 的 named_parameters 接收 |
| process_weights_after_loading | (已跑过) | **再跑一次** → 二次 reshape |

#### 2.1.2 哪些量化方法会触发

以下量化方法在 `process_weights_after_loading` 中**改变 weight / weight_scale 的 shape 或数值**，会导致 sender 发出的数据与 receiver 期望的 buffer 不匹配：

**FP8 (基础)** — `vllm_ascend/quantization/methods/fp8.py:60-76`：

```python
def process_weights_after_loading(self, layer):
    # 修改 weight_scale 数值 + 维度
    layer.weight_scale.data = layer.weight_scale.data.view(torch.int32) >> 23 & 0xFF
    n_dim, k_dim = layer.weight_scale.data.shape
    layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)  # 2D → 3D
    layer.weight.data = layer.weight.data.transpose(0, 1)                             # weight 转置
    layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)
```

**FP8 ds_linear (DS 原生 block-wise)** — `vllm_ascend/quantization/methods/w8a8_dynamic.py:60-77`：

```python
def process_weights_after_loading(self, layer):
    layer.weight_scale.data = layer.weight_scale.data.view(torch.int32) >> 23 & 0xFF
    layer.weight_scale.data = layer.weight_scale.data.to(torch.uint8)
    layer.weight_scale.data = layer.weight_scale.data.repeat_interleave(4, dim=1).repeat_interleave(128, dim=0)
    n_dim, k_dim = layer.weight_scale.data.shape
    layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)  # → 3D
    layer.weight.data = layer.weight.data.transpose(0, 1)
    layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)
```

**W4A4 FlatQuant** — `vllm_ascend/quantization/methods/w4a4_flatquant.py:140-156`：在 forward 中 `x.view(-1, left_dim, right_dim)`，weight 本身也是 3D 形态。

**W4A4 mxfp4 / W4A8 mxfp4** — `vllm_ascend/quantization/methods/w4a4_mxfp4.py` 等：`npu_format_cast` + `transpose(1, 2)`。

**W8A8 static + NZ 转换** — `vllm_ascend/quantization/methods/w8a8_static.py:76-91`：

```python
def process_weights_after_loading(self, layer):
    ...
    layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
    layer.weight.data = maybe_trans_nz(layer.weight.data)  # NZ 格式转换
    layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
    layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
```

**W8A8 MXFP8 DS moe** — `vllm_ascend/quantization/methods/w4a8_mxfp4.py` 中 `process_weights_after_loading`：

```python
g, n, k = layer.w13_weight_scale.shape
layer.w13_weight_scale.data = (
    layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)  # → 4D
)
```

#### 2.1.3 后果

- **Shape 不匹配的 recv**：sender `param.contiguous()` 后元素数与 receiver `param.numel()` 相同，recv 表面成功，但数据**语义完全错乱**（被 transpose / reshape 过）。
- **数值被重复处理**：receiver 把"已经被 `>> 23 & 0xFF` 解码过的 scale"当作原始 FP8 数据再走一次 `view(int32) >> 23 & 0xFF`，数值彻底坏掉。
- **Receiver 再跑 process**：netloader.py L321-322 标记了 `need_process_weights_after_loading = True`，必然再跑一次 `process_weights_after_loading`，对已经损坏的数据二次 reshape。
- **现象**：加载日志正常，一推理就崩溃或输出乱码 —— 与用户描述的 dsv4 表现一致。

> **这就是用户描述"dsv4 不行，量化变成三维，加载能做但推理崩溃"的代码层面根因。**

#### 2.1.4 反向场景同样不可行

直觉上可以想让 sender "跳过 process 直接发 raw 权重"，但当前架构不支持：

- Sender 自身也是 netloader 加载流程，没法跳过自己的 process（否则 sender 自己都跑不起来）
- 即使有 sender 模式开关，process 的副作用（派生 param、NZ 转换）发生在 model 对象上，没法局部撤销后再发送

---

### 2.2 【严重】缺陷 2：参数顺序匹配完全靠默契，无协议协商

`P2PSend` 和 `P2PLoad` 各自独立遍历 `model.named_parameters()`，**没有 manifest 交换**。下列任何一种情况都会让顺序错位，后续所有 recv 全部张冠李戴：

#### 2.2.1 `aclnn_input_scale` 单边跳过

`P2PSend` 跳过名字含 `aclnn_input_scale` 的 param，`P2PLoad` 不跳过。

当前能工作**仅因为** `aclnn_input_scale` 是 `process_weights_after_loading` 阶段创建的派生 param（见 `vllm_ascend/_310p/quantization/methods/w8a8_static.py:80-91`），receiver 在 elastic_load 阶段还没这个 param。

但只要某个量化方法在 `initialize_model` 阶段就预创建了相关 param（lazy 策略或新量化方法），两边参数列表立刻长度不同 → 错位。

#### 2.2.2 标量 param 单边跳过

Receiver 跳过 `len(param.shape) == 0`，sender 不跳过。

某些 per-tensor scale 会存成 0-d tensor（`torch.empty(1)` 与 `torch.tensor(0.5)` 行为不同），如果未来出现真正的标量 Parameter，sender 发了 receiver 不收 → 错位 → 死锁或张冠李戴。

#### 2.2.3 派生 Parameter 数量差异

`process_weights_after_loading` 在不同量化方法里创建的派生 param 数量和名字都不同：

- `aclnn_input_scale` / `aclnn_input_scale_reciprocal` / `aclnn_input_offset`
- `deq_scale`
- `weight_scale_fp32`
- `weight_1` / `weight_2` / `weight_1_scale` / `weight_2_scale`（chunked，见 `w8a8_dynamic.py:103-115`）

Sender 已有这些派生 param，receiver 还没有 → 参数列表长度不同 → recv 错位。

---

### 2.3 【严重】缺陷 3：主模型与 MTP（draft）几乎不区分

#### 2.3.1 现状代码

`netloader.py:134-137` 的 draft 检测：

```python
@staticmethod
def _is_draft_model(model_config: ModelConfig) -> bool:
    return getattr(model_config, "runner_type", None) == "draft"
```

Draft 分支处理（`netloader.py:198-212`）**只做了一件事** —— 把 source 列表里的端口统一加 `DRAFT_PORT_OFFSET = 10000`：

```python
sources = self.source
if is_draft:
    sources = [
        {
            "device_id": s["device_id"],
            "sources": [
                f"{parts[0]}:{int(parts[1]) + DRAFT_PORT_OFFSET}"
                for addr in s.get("sources", [])
                ...
            ],
        }
        for s in self.source
        ...
    ]
```

#### 2.3.2 问题

- `source`、`int8_cache`、`int8_cache_name`、`output_prefix` **全部共用同一份配置**，没有 per-model 通道。
- 量化配置走 `vllm_config`，speculative decoding 框架会给 draft 配独立的 `model_config` 和 `quant_config`，但 NetLoader **没有读取 draft 的 quant_config 做差异化处理**。
- 若主模型是 W8A8（int8 weight + `aclnn_input_scale`），MTP 是 FP8 或非量化：
  - Receiver 加载 MTP 时按 MTP 的 quant_config 初始化 param
  - Sender（同机 rank 0）发的却是主模型 quant 配方下的权重
  - 参数名 / shape / dtype 全错位，触发缺陷 2 的连锁反应
- `int8_cache_name` 是单个正则 list，**没法表达"这些名字属于 target，那些属于 draft"**。
- 多级 MTP 或 EP + draft 组合下，单个 `DRAFT_PORT_OFFSET = 10000` 也不够。

---

### 2.4 【中等】缺陷 4：`int8_cache` 是 sender 单边优化，与 receiver 无协议

`ElasticServer.__init__`（`interaction/elastic.py`）在 sender 侧按 `int8_cache_name` 正则匹配 param 名，把 int8 param 备份到 HBM 或 DRAM：

```python
int8_pattern = "|".join(map(re.escape, int8_cache_name)) if int8_cache_name is not None else "(?:)"
for name, param in self.model.named_parameters():
    if param.dtype == torch.int8:
        if int8_cache == "hbm":
            if int8_cache_name is None or re.search(int8_pattern, name) is not None:
                self.original_int8[name] = param.data.clone().detach()
        elif int8_cache == "dram":
            ...
```

**问题**：

- 用正则匹配 param 名，匹配错就漏备份或错备份，调试困难。
- Receiver 完全不知道哪些是 cached int8，**只看 sender 给的 dict key**。
- 如果 sender 备份的是 NZ 转换前的 int8（例如 process 之前的 raw），但 receiver 期望的是 NZ 转换后的格式，recv 会写入错误数据。
- 这个机制**绑定 ascend 自家的 W8A8 / W8A8S 命名约定**，跨量化方法无通用性。

---

### 2.5 【中等】缺陷 5：`revert_to_default` 的 quant 分支依赖隐式时序

`netloader.py:330-369` 的 `revert_to_default`：

```python
def revert_to_default(self, model_config, vllm_config, device_config, prefix=""):
    load_config = deepcopy(self.load_config)
    load_config.model_loader_extra_config = {}
    load_config.load_format = "auto"
    default_model_loader = DefaultModelLoader(load_config)

    if model_config.quantization is None:
        model = default_model_loader.load_model(...)
        need_process_weights_after_loading = False
    else:
        # 量化场景下手动 initialize_model + load_weights
        need_process_weights_after_loading = True
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(...)
            default_model_loader.load_weights(model, model_config)
        model = model.eval()
    return model, need_process_weights_after_loading
```

**问题**：

- 这个函数不能独立复用 —— 它依赖调用方检查 `need_process_weights_after_loading` 标记并在外部跑 `process_weights_after_loading`（`netloader.py:321-322`）。
- 在 elastic_load 失败的回退路径（L227-245）中，虽然 `_quant_config` 和 `model_config_backup` 被恢复，但 **`initialize_model` 已经创建的派生 param 残留没被清理**，再走 default loader 时可能带着脏 param。

---

### 2.6 【轻】缺陷 6：source 过滤与 device_id 假设

`elastic_load`（`load.py`）按 `s["device_id"] == device_id` 过滤 source，`device_id = torch.distributed.get_rank()`（`netloader.py:161`）。

**问题**：

- 用 global rank 作为 device_id 索引，假设 source 配置里的 device_id 也用同一套 rank 编号。
- 多机或 EP 场景下，rank 编号体系可能不一致（例如 rank 0 在 node 0，但 source 配置里用 local device id）。
- Draft 的 source list 通过端口偏移复用同一份配置，**没法表达"draft 用另一组 device"**。

---

## 3. 问题回答矩阵

| 用户痛点 | 对应缺陷 | 代码根因 |
|---|---|---|
| 不同模型量化不同，主模型 / MTP 量化不同，NetLoader 怎么区分 | 缺陷 3 | 只看 `runner_type=="draft"` 做端口偏移；quant_config / int8_cache 共用，无 per-model 通道 |
| dsv4 不行 | 缺陷 1 | DS 原生 FP8 process 后 weight_scale 变 3D（`w8a8_dynamic.py:65`、`fp8.py:65`） |
| d2d 加载能做但推理崩溃 | 缺陷 1 + 2 | Sender 已 processed 的 3D 权重 contiguous 后塞进 receiver 2D buffer；receiver 再 process 一次 → 数值坏掉或 shape 错 |
| 量化变成三维 | 缺陷 1 | FP8 / MXFP8 / W4A4 的 `process_weights_after_loading` 都会 reshape 到 3D / 4D |
| 功能变成 model specific | 缺陷 1 + 2 + 3 + 4 | 当前实现仅对"process_weights_after_loading 不改 weight shape"的量化方法安全；任何 reshape / transpose / NZ 转换 / 派生 param 的量化都会破坏 sender-receiver 默认契约 |

### 3.1 当前 NetLoader **能工作** 的窄场景

满足以下**全部**条件的模型才能用：

- 主模型与 MTP 量化配置完全一致
- 量化方法的 `process_weights_after_loading` **不改变 weight / weight_scale 的 shape**
- 不依赖 NZ 格式转换（`maybe_trans_nz`）
- 不创建会被 receiver 遗漏的派生 Parameter
- 典型例子：纯 per-tensor FP8（`weight_scale` 保持 `[1]`）、部分 W8A8 static 路径（无 NZ）

### 3.2 当前 NetLoader **会崩** 的场景

只要命中下列**任一**条件，NetLoader 就会出问题：

- DSv4 / 任何 block-wise FP8（weight_scale 变 3D）
- W4A4 FlatQuant（weight 变 3D）
- W4A4 / W4A8 MXFP4（`npu_format_cast` + transpose）
- W8A8 static 开启 NZ 转换（`maybe_trans_nz`）
- DS W8A8 MXFP8 moe（weight_scale 变 4D）
- 主模型与 MTP 量化配置不同的任意组合

---

## 4. 信息缺口（设计修复方案前需要补齐）

1. **Sender 侧 process 完成时机** —— 已确认：`process_weights_after_loading` 在 `BaseModelLoader.load_model` 内部（`vllm/model_executor/model_loader/base_loader.py:80`）执行，先于 `load_model` 返回。Sender 启动 `ElasticServer` 时权重确实已 processed。
2. **Sender 是否知道自己被作为 d2d 源** —— 当前代码里 sender 也是普通 netloader 加载流程，只是恰好某个 rank 的 `source` 配置指向自己。没有"sender 专属"的 process 跳过开关。
3. **Receiver 是否有办法从 sender 拉取 manifest** —— 当前协议层没有这个能力，需要协议升级。
4. **Draft 的 quant_config 是否被 vLLM 上游正确传递** —— 需要看 vLLM 上游 speculative worker 的调用链（`vllm/spec_decode/` 和 `vllm/draft/`）确认。

---

## 5. 修复方向建议（不在本文范围）

作为分析结论的自然延伸，记录三个层次的思路：

### 5.1 短期 workaround：白名单

在 `netloader.py:225` 处加条件，仅当当前量化方法属于"安全白名单"时才走 `elastic_load`；否则直接 fallback 到 `revert_to_default`。白名单可以基于：

- `quant_config.quant_method` 的方法名
- 量化方法的 `process_weights_after_loading` 是否 reshape（需要每个 ascend quant method 标注）

优点：改动小，立竿见影。
缺点：白名单维护成本高，新量化方法容易漏登。

### 5.2 中期协议升级：交换 manifest

d2d 传输前先通过 socket 协商交换 manifest：

```python
# 协议示例
manifest = {
    "model_id": "...",       # 区分主模型 / draft
    "quant_method": "fp8",
    "processed": True,       # sender 是否已 processed
    "params": [
        {"name": "weight", "shape": [N, K], "dtype": "float8_e4m3fn"},
        {"name": "weight_scale", "shape": [N, K//2, 2], "dtype": "uint8"},
        ...
    ]
}
```

Receiver 收到 manifest 后决定：

- shape 匹配 → 正常 recv
- shape 不匹配但 numel 匹配 → 警告 + 跳过本地 process
- shape 完全不匹配 → fallback 到 default loader

优点：协议层根治，新量化方法自动兼容。
缺点：协议改动大，需要同步升级 sender 和 receiver。

### 5.3 长期架构：拷贝 safetensors 原始权重

把 NetLoader 从"拷贝已 processed 权重"改为"拷贝 safetensors 原始权重 + 让 receiver 完整走 process"。具体做法：

- Sender 把加载过的 safetensors 文件路径 + 偏移信息通过 d2d 发给 receiver
- Receiver 用这些信息直接从共享存储（或 sender 通过 RDMA 共享的内存）读取原始权重
- Receiver 完整走 `DefaultModelLoader.load_weights` + `process_weights_after_loading`

优点：天然兼容所有量化方法，与上游协议一致。
缺点：receiver 侧的 process 开销没法省，d2d 的性能优势被削弱。

---

## 6. 关键文件清单

修改 / 参考时必看：

- `vllm-ascend/vllm_ascend/model_loader/netloader/netloader.py`：NetLoader 主逻辑
- `vllm-ascend/vllm_ascend/model_loader/netloader/executor/elastic_load.py`：d2d 协议（P2PSend / P2PLoad）
- `vllm-ascend/vllm_ascend/model_loader/netloader/interaction/elastic.py`：ElasticServer / ElasticClient
- `vllm-ascend/vllm_ascend/quantization/methods/fp8.py`：典型 reshape 量化
- `vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py`：DS 原生 FP8（dsv4 使用）
- `vllm-ascend/vllm_ascend/quantization/methods/w8a8_static.py`：W8A8 + NZ
- `vllm-ascend/vllm_ascend/quantization/methods/w4a4_flatquant.py`：3D weight
- `vllm-ascend/vllm_ascend/quantization/methods/w4a8_mxfp4.py`：moe weight_scale reshape 到 4D
- `vllm/vllm/model_executor/model_loader/utils.py`：上游 `process_weights_after_loading` 入口
- `vllm/vllm/model_executor/model_loader/base_loader.py`：上游 `BaseModelLoader.load_model`（调用 pwal 的位置）

---

## 附录 A：验证手段

本文是纯静态分析，验证手段：

1. **重新读取关键文件**确认 file:line 引用准确（已在 `executor/elastic_load.py`、`netloader.py`、`quantization/methods/*.py`、上游 `model_loader/utils.py` 之间交叉验证）
2. **对照 vLLM 上游** `process_weights_after_loading` 的调用点（`base_loader.py:80`），确认 sender 侧 process 完成早于 `ElasticServer.start()`
3. **缺陷 1 的现场验证**：找一个 DSv4 FP8 checkpoint，在 sender 侧 `ElasticServer.__init__` 之前打印 `weight_scale.shape`，确认是 3D（这一步需要在 NPU 环境跑，不在本文范围）

## 附录 B：术语表

- **d2d**：device-to-device，指 NPU 之间直接通过 HCCL 传权重，绕过 CPU 内存中转
- **HCCL**：Huawei Collective Communication Library，华为集合通信库，类似 NCCL
- **NZ 格式**：Ascend NPU 的一种分块矩阵内存布局，对 GEMM 加速有利，但与普通 row-major 不兼容
- **MTP**：Multi-Token Prediction，一种 speculative decoding 方案，draft model 与主模型量化配置可能不同
- **process_weights_after_loading**：vLLM 上游约定，权重加载后由量化方法执行的 post-processing 钩子，用于 reshape、repacking、scale 计算等
