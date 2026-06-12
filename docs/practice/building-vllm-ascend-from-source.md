# vllm-ascend 从源码编译安装

> 本文档指导用户在 Ascend NPU 环境下从源码编译安装 vllm 和 vllm-ascend。涵盖 **华为内网环境**（需配置内部 PyPI 源）和 **外网环境** 两种场景，用户可根据实际网络条件选择对应步骤。安装前请确保已准备好 CANN、torch_npu 等依赖。

## 仓库地址

```text
# 官方仓库
https://github.com/vllm-project/vllm.git
https://github.com/vllm-project/vllm-ascend.git

# Fork 仓库（例）
https://github.com/xuchi-0808/vllm.git
https://github.com/xuchi-0808/vllm-ascend.git
```

## 1. 安装 [vllm](https://github.com/vllm-project/vllm)

```bash
git clone <vllm-repo> vllm
cd vllm
```

```bash
VLLM_TARGET_DEVICE=empty pip install -v -e . --no-build-isolation
```

> `VLLM_TARGET_DEVICE=empty` 必须设置，否则会尝试编译 CUDA kernel 而报错。
>
> `--no-build-isolation` 避免重新构建 build 依赖（如 setuptools、wheel 等），可大幅减少安装时间。

## 2. 安装 [vllm-ascend](https://github.com/vllm-project/vllm-ascend)

### 内网环境

无法访问 PyPI 时需要配置内部 triton 源：

```bash
git clone <vllm-ascend-repo> vllm-ascend
cd vllm-ascend
```

```bash
pip install -v -r requirements.txt \
  --trusted-host triton-ascend.osinfra.cn \
  --upgrade-strategy only-if-needed \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
pip install -v -e . \
  --no-build-isolation \
  --trusted-host triton-ascend.osinfra.cn \
  --upgrade-strategy only-if-needed \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

### 外网环境

```bash
git clone <vllm-ascend-repo> vllm-ascend
cd vllm-ascend
```

```bash
pip install -v -r requirements.txt
pip install -v -e . --no-build-isolation
```

## 3. FAQ

### 编译时报 `cp: cannot create regular file '/mc2/...'`

错误：`cp: cannot create regular file '/mc2/moe_dispatch_normal/op_kernel/utils/': No such file or directory`

原因：`csrc/build_aclnn.sh` 中使用了未定义的变量 `$SCRIPT_DIR`，应改为 `$ROOT_DIR`。

```bash
sed -i 's/\$SCRIPT_DIR/\$ROOT_DIR/g' csrc/build_aclnn.sh
```

```bash
pip install -v -e . --no-build-isolation
```

### 编译时报 `Stale file handle` 错误

错误：

```text
rm: cannot remove 'build/prepare_build/.../proto_stub.cpp': Stale file handle
rm: fts_read failed: Stale file handle
```

原因：NFS 上上一次编译残留的文件句柄过期。

解决：

```bash
rm -rf csrc/build build
pip install -v -e . --no-build-isolation
```

### vllm 版本号显示 `dev` 不是期望的版本号

原因：vllm 使用 `setuptools_scm` 从 git tag 获取版本号，当前分支没有对应 tag 时默认为 `"dev"`。某些代码（如 `patch_mla_prefill_backend.py`）依赖 `vllm.__version__` 做版本判断，`"dev"` 会导致行为异常。

解决：拉取目标 tag，或在 `pip install` 前设置 `SETUPTOOLS_SCM_PRETEND_VERSION`。

### CPack 打包报错 `No such file: CANN-custom_ops*.run`

错误：

```text
CPack Error: Problem running install command: cmake --build . --target "preinstall"
csrc/build_aclnn.sh: line 84: ./output/CANN-custom_ops*.run: No such file or directory
```

原因：前一次编译残留的 `csrc/build` 目录导致 CPack 打包不一致。

解决：

```bash
rm -rf csrc/build csrc/output
pip install -v -e . --no-build-isolation
```

### `HAS_TRITON=False` 导致自定义算子未注册

现象：启动 server 时 torch.compile 阶段报 `AttributeError: '_OpNamespace' 'vllm' object has no attribute 'qkv_rmsnorm_rope'`。

原因：`vllm_ascend/ops/__init__.py` 中 `if HAS_TRITON:` 依赖 vllm 的 `triton_utils.HAS_TRITON`。该值通过 `import triton.backends` 判断，某些 triton-ascend 版本未暴露 `triton.backends` 模块，导致 `HAS_TRITON=False`，相关自定义算子被跳过注册，后续 NPU 编译 pass 调用时找不到。

临时绕过：

```bash
sed -i 's/if HAS_TRITON:/if True:  # force for bisect/' vllm_ascend/ops/__init__.py
pip install -v -e . --no-build-isolation
```

> 此修改仅适用于验证场景，正式修复需确保 triton-ascend 版本能正确导出 `triton.backends`。

---

## 4. 验证安装

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import vllm_ascend; print(vllm_ascend.__version__)"
```
