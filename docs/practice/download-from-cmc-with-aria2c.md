# 使用 aria2c 从 CMC 下载安装包

## 适用场景

在 NPU 服务器上下载 CMC 分发的安装包（如 MindIE、CANN、驱动等）。推荐使用 aria2c 支持断点续传和多线程加速。

---

## 第一步：安装 aria2c

### 下载二进制包

从 aria2 静态构建发布页下载 aarch64 版本：

[https://github.com/abcfy2/aria2-static-build/](https://github.com/abcfy2/aria2-static-build/releases)

> 如果服务器 CPU 架构不是 aarch64，在上述发布页选择对应架构的包。NPU 服务器一般为 aarch64。

### 解压并安装

```bash
unzip aria2-aarch64-linux-musl_static.zip
sudo cp aria2c /usr/local/bin/
```

### 验证

```bash
aria2c --version
```

正常输出类似：

```text
aria2 version 1.37.0
...
```

---

## 第二步：从 CMC 获取下载链接

1. 在浏览器打开 CMC 页面，找到需要下载的安装包
2. 点击下载按钮，浏览器弹出下载模块
3. **在下载弹窗上右键**，选择「复制链接地址」（不同浏览器措辞略有差异）
4. 此时下载链接已在剪贴板中，可用于后续在服务器上下载

---

## 第三步：在服务器上下载

使用 aria2c 下载，开启多线程和断点续传：

```bash
aria2c -c -x 16 -s 16 --check-certificate=false '粘贴这里复制的链接'
```

参数说明：

| 参数                        | 作用                                        |
| --------------------------- | ------------------------------------------- |
| `-c`                        | 断点续传，中断后重新执行相同命令会继续下载  |
| `-x 16`                     | 同一服务器最多开启 16 个连接                |
| `-s 16`                     | 文件分 16 片并行下载                        |
| `--check-certificate=false` | 跳过 SSL 证书验证（CMC 链接可能证书不匹配） |

### 注意事项

- **使用前需确保服务器已配置外网代理**，否则 aria2c 无法访问 GitHub 和 CMC 的下载地址。代理配置方式待后续文档补充。
- 如果下载中断，**重新执行同样的命令**，`-c` 参数会自动续传，无需重新开始。

---

## 参考

- [aria2 官方文档](https://aria2.github.io/manual/en/html/)
- [abcfy2/aria2-static-build 发布页](https://github.com/abcfy2/aria2-static-build/releases)
