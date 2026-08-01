# vllm-ascend 从源码编译安装

> 本文档指导用户在 **Docker 容器** 中从源码编译安装 vllm 和 vllm-ascend。涵盖 Y/G 区（内网）、B 区（受限外网）、外网三种网络环境。裸机环境可参考但未经验证。
>
> 安装前请确保容器内已准备好 CANN、torch_npu 等依赖。
>
> 也推荐你将本文直接粘贴给大模型，让大模型指导你进行安装，可以省去查找 FAQ 的麻烦。

## 0. vllm-ascend 与 vllm 版本配套

vllm-ascend 依赖特定版本的 vllm，**推荐安装配套版本**，否则可能出现接口不兼容。查找方法：

- **新版**（2026-06-10 之后）：读 `.github/vllm-release-tag.commit`，内容为 vllm 的 git tag
- **旧版**（2026-06-10 之前）：读 `docs/source/conf.py`，在 `myst_substitutions` 字典里找 `main_vllm_commit`（精确 hash）和 `main_vllm_tag`

```bash
# 新版
cat .github/vllm-release-tag.commit

# 旧版
grep -E 'main_vllm_(commit|tag)' docs/source/conf.py
```

> 官方 Docker 镜像通常已预装匹配版本，无需手动查找。仅在源码安装或 bisect 切换版本时需要。

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
# Docker 镜像通常预装了 vllm 和 vllm-ascend，先卸载再装指定版本
pip uninstall vllm vllm-ascend -y

VLLM_TARGET_DEVICE=empty pip install -v -e . --no-build-isolation
```

> `VLLM_TARGET_DEVICE=empty` 必须设置，否则会尝试编译 CUDA kernel 而报错。
>
> `--no-build-isolation` 避免重新构建 build 依赖（如 setuptools、wheel 等），可大幅减少安装时间。

## 2. 安装 [vllm-ascend](https://github.com/vllm-project/vllm-ascend)

### Y/G 区环境（内网服务器）

无法访问 PyPI，使用内部 triton 源：

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

### B 区环境（受限外网）

无法访问官方 PyPI，但可访问阿里云镜像：

```bash
git clone <vllm-ascend-repo> vllm-ascend
cd vllm-ascend
```

```bash
pip install -v -r requirements.txt \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com
pip install -v -e . \
  --no-build-isolation \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com
```

> 其他镜像（华为云、清华）经测试存在 aarch64 包缺失或 403 问题，不推荐。

### 外网环境（可以访问全球互联网）

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

### Docker 镜像预装 vllm 导致安装失败

现象：`pip install -e .` 后 import 的仍是旧版本，或安装过程报版本冲突。

原因：Docker 镜像通常预装了 vllm，pip 认为已满足依赖而跳过安装。

解决：先卸载再装。

```bash
pip uninstall vllm -y
VLLM_TARGET_DEVICE=empty pip install -v -e . --no-build-isolation
```

### 缺少 `setuptools-rust`

现象：编译时报错找不到 rust 工具链。

原因：部分 Docker 镜像未预装 `setuptools-rust`。

解决：

```bash
pip install setuptools-rust
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
