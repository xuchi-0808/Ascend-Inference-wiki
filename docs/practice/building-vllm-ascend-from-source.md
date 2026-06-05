# vllm-ascend 从源码编译安装

## 1. 安装 [vllm](https://github.com/vllm-project/vllm)

```bash
git clone <vllm-repo>
cd vllm

VLLM_TARGET_DEVICE=empty pip install -e . -v
```

> `VLLM_TARGET_DEVICE=empty` 必须设置，否则会尝试编译 CUDA kernel 而报错。

## 2. 安装 [vllm-ascend](https://github.com/vllm-project/vllm-ascend)

### 内网环境

无法访问 PyPI 时需要配置内部 triton 源：

```bash
git clone <vllm-ascend-repo>
cd vllm-ascend

pip install -r requirements.txt \
  --trusted-host triton-ascend.osinfra.cn \
  --upgrade-strategy only-if-needed \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple

pip install -e . \
  --trusted-host triton-ascend.osinfra.cn \
  --upgrade-strategy only-if-needed \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

### 外网环境

```bash
git clone <vllm-ascend-repo>
cd vllm-ascend
pip install -r requirements.txt
pip install -e .
```

## 3. FAQ

### 编译时报 `cp: cannot create regular file '/mc2/...'`

错误：`cp: cannot create regular file '/mc2/moe_dispatch_normal/op_kernel/utils/': No such file or directory`

原因：`csrc/build_aclnn.sh` 中使用了未定义的变量 `$SCRIPT_DIR`，应改为 `$ROOT_DIR`。

```bash
sed -i 's/\$SCRIPT_DIR/\$ROOT_DIR/g' csrc/build_aclnn.sh
pip install -e .
```

### vllm 版本号显示 `dev` 不是期望的版本号

原因：vllm 使用 `setuptools_scm` 从 git tag 获取版本号，当前分支没有对应 tag 时默认为 `"dev"`。某些代码（如 `patch_mla_prefill_backend.py`）依赖 `vllm.__version__` 做版本判断，`"dev"` 会导致行为异常。

解决：拉取目标 tag，或在 `pip install` 前设置 `SETUPTOOLS_SCM_PRETEND_VERSION`。

---

## 4. 验证安装

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import vllm_ascend; print(vllm_ascend.__version__)"
```
