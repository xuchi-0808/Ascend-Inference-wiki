---
date: 2026-06-06
tags:
  - 环境
  - troubleshooting
  - 双机
  - 编译
---

# vllm-ascend 双机环境调试踩坑记录

vllm-ascend 双机 nightly 测试（Qwen3/Qwen 系列 MoE 模型）在环境准备和手动拉起过程中踩过的坑，按环节分类记录。

---

## 1. 编译安装

### 1.1 build_aclnn.sh 变量未定义

**错误**：

```text
cp: cannot create regular file '/mc2/moe_dispatch_normal/op_kernel/utils/': No such file or directory
```

**原因**：`csrc/build_aclnn.sh` 中使用了未定义的变量 `$SCRIPT_DIR`，实际应为 `$ROOT_DIR`。

**解决**：

```bash
sed -i 's/\$SCRIPT_DIR/\$ROOT_DIR/g' csrc/build_aclnn.sh
```

然后重新 `pip install -e .`。

*注：该 bug 在 vllm-ascend `releases/v0.20.2rc` 及之前版本存在。*

---

### 1.2 NFS Stale file handle

**错误**：

```text
rm: cannot remove 'build/prepare_build/.../proto_stub.cpp': Stale file handle
rm: fts_read failed: Stale file handle
Error running build_aclnn.sh
```

**原因**：NFS 文件系统上编译中断后残留的文件句柄过期。再次编译时 `rm` 读取失效句柄，抛出 `Stale file handle`。

**解决**：

```bash
rm -rf csrc/build build
pip install -v -e . --no-build-isolation
```

**预防**：若 `pip install -e .` 被中断（Ctrl+C 或超时），先清 `csrc/build` 和 `build` 再重试。

---

### 1.3 vllm 版本号 `dev` 导致 `patch_mla_prefill_backend.py` 崩溃

**错误**：

```text
ModuleNotFoundError: No module named 'vllm.v1.attention.backends.mla.prefill'
```

**原因**：vllm 使用 `setuptools_scm` 从 git tag 读取版本号。fork 仓库的分支上没有 `v0.20.2` tag，`vllm.__version__` 返回 `"dev"`。`patch_mla_prefill_backend.py` 中：

```python
if not vllm_version_is("0.20.2"):
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
```

`"dev" != "0.20.2"` → 执行 import → 该模块在 v0.20.2 中不存在 → ModuleNotFoundError。

**解决**：

```bash
# 推荐：直接从 upstream clone
git clone https://github.com/vllm-project/vllm.git -b releases/v0.20.2
cd vllm && pip install -e .
```

**其他方案**：

```bash
# 切到 tag
git checkout v0.20.2 && pip install -e .

# 或设环境变量伪造版本号
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2 pip install -e vllm
```

---

## 2. 双机 vllm serve 启动

### 2.1 Node 1 server 秒退 — `hang_until_terminated` 竞态

**现象**：Node 1 的 pytest 17 秒就 `1 passed`，Node 0 等待 54 分钟后报 `Engine core initialization failed. Failed core proc(s): {}`。

**原因**：Node 1 的 `hang_until_terminated()` 方法轮询 Node 0 的 `/health`。当 Node 0 的 HTTP 服务尚未就绪时，连接错误被 `except Exception: break` 捕获，Node 1 立即退出并杀死 server 进程。

**修复**：在 `conftest.py` 的 `hang_until_terminated` 中，先等待 leader 的 `/health` 返回 200，再进入原有轮询逻辑：

```python
# 修复前 — 首次连接失败就 break
except Exception:
    break

# 修复后 — 先循环等 leader 就绪
while True:
    try:
        resp = client.get(url, timeout=5)
        if resp.status_code == 200:
            break  # leader 就绪
    except Exception:
        pass  # 继续等
    time.sleep(5)
```

---

### 2.2 需要设置 16 个环境变量

**原因**：测试框架的 `RemoteOpenAIServer._start_server()` 会通过 `env_dict.update(env)` 自动注入 16 个环境变量，手动 `vllm serve` 时需全部由用户自行设置。

**分布**：

| 来源 | 数量 | 变量名 |
|:---|:---:|:---|
| yaml `env_common` | 8 | `HCCL_OP_EXPANSION_MODE`, `TASK_QUEUE_ENABLE`, `OMP_PROC_BIND`, `OMP_NUM_THREADS`, `PYTORCH_NPU_ALLOC_CONF`, `HCCL_BUFFSIZE`, `SERVER_PORT`, `NUMEXPR_MAX_THREADS` |
| `DistEnvBuilder.build()` | 7 | `HCCL_IF_IP`, `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `LOCAL_IP`, `NIC_NAME`, `MASTER_IP` |
| `conftest.py` `_start_server()` | 1 | `VLLM_WORKER_MULTIPROC_METHOD="spawn"` |

**解决**：详见 `vllm_serve_双机手动拉起.md` 中的完整脚本。

---

### 2.3 网卡名需要手动确认

`DistEnvBuilder` 自动检测网卡名。手动拉时需要确认：

```bash
python3 -c "import psutil,socket; [print(i) for i,a in psutil.net_if_addrs().items() for x in a if x.family==socket.AF_INET and x.address=='<本机IP>']"
```

影响 4 个变量：`HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `NIC_NAME`。

---

## 3. Benchmark 测试

### 3.1 `_custom.py` 配置文件不存在

**现象**：ais_bench 找不到 `vllm_api_general_chat_custom` 配置。

**原因**：ais_bench 包中只有模板 `vllm_api_general_chat.py`（含占位符）。`_custom` 后缀的文件由测试框架运行时生成，包中不存在。

**解决**：手动创建自定义配置文件，详见 `vllm_serve_双机手动拉起.md` 中 benchmark 一节。

---

### 3.2 `BENCHMARK_HOME` 指向错误

**错误**：

```text
cp: cannot copy a directory, '.../gsm8k', into itself, '.../gsm8k/gsm8k'
```

**原因**：`BENCHMARK_HOME` 设为了 ais_bench 包内路径，导致 `dataset_path_local` 和 `DATASET_DIR` 指向同一目录，`cp -r` 自己拷自己。

**解决**：`BENCHMARK_HOME` 设为独立工作目录：

```bash
export BENCHMARK_HOME="/tmp/aisbench_work"
```

---

### 3.3 `CONFIG_YAML_PATH` 默认值不对

**现象**：pytest 加载了 `DeepSeek-V3.yaml` 而非 Qwen3 配置。

**原因**：`MultiNodeConfigLoader.DEFAULT_CONFIG_NAME = "DeepSeek-V3.yaml"`。CI 通过 `run.sh` 传递该变量，手动跑 pytest 时默认值不匹配。

**解决**：

```bash
export CONFIG_YAML_PATH="Qwen3-235B-A22B.yaml"
```

---

### 3.4 `pytest-asyncio` 未安装

**现象**：测试报告 `1 skipped, 20 warnings`，async 测试未执行。

**原因**：`test_multi_node.py` 使用 `@pytest.mark.asyncio`，缺少 `pytest-asyncio` 插件。

**解决**：

```bash
pip install pytest-asyncio
```

---

### 3.5 modelscope 证书校验失败（内网环境）

**错误**：

```text
SSLCertVerificationError: certificate verify failed: self-signed certificate in certificate chain
```

**解决**：

```bash
# 方案 A：确认 no_proxy 配置
export no_proxy=127.0.0.1,localhost,.local,.huawei.com

# 方案 B：使用本地数据集（推荐）
# 在 yaml benchmark 的 acc 段加：
dataset_path_local: /path/to/local/gsm8k
```

---

## 4. 资源竞争

### 4.1 被其他用户抢卡

**现象**：Node 1 的 `vllm serve` 启动后立刻崩溃，无日志。

**排查**：

```bash
npu-smi info
```

找到占卡用户后协调释放。

---

## 5. 本次调试环境

| 组件 | 版本 / 分支 |
|:---|:---|
| vllm | `releases/v0.20.2`（tag: `v0.20.2`） |
| vllm-ascend | `A00275` 分支，基于 5/22 commit `68a4db55` |
| CANN | cann-9.0.0 |
| npu-smi | 25.5.2 |
| 容器镜像 | `nightly-ci-main-a3` |
| ais_bench | 预装在容器镜像中 |
