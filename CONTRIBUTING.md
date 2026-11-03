# 贡献指南

欢迎为 Ascend Inference Wiki 贡献内容。本指南帮你快速判断「文档放哪、怎么写、怎么提」。

## 仓库定位

面向社区与 AI Agent 的昇腾大模型推理知识库。内容灵活、限制小，覆盖 AI Infra、推理特性、模型知识、实践经验。区别于官方文档，这里欢迎经验总结、个人见解、复盘叙事。

## 1. 判断文档归属

仓库按**读者意图**分五大类。写完一篇，问自己「读者打开这页想干嘛」：

| 分类 | 读者来 | 放什么 | 例子 |
| --- | --- | --- | --- |
| 操作指南 | 做 | 跟着步骤完成任务 | 编译安装、部署拉起、跑 benchmark |
| 参考手册 | 查 | 查阅事实 | 模型档案、算子清单、配置参数、术语表 |
| 原理解析 | 懂 | 理解机制与取舍 | MLA 原理、并行策略对比、源码追踪 |
| 设计方案 | 议 | 方案/RFC/决策 | 架构设计、技术提案、ADR |
| 博客 | 看 | 叙事/复盘/观点 | postmortem、调优日志、版本解读 |

口诀：**做、查、懂、议、看**。

### 主题用 tag，不用目录

文档的技术主题（DeepSeek、双机、精度）不要做成目录，用 frontmatter 的 `tags` 表达。一篇文档目录只归一次，但可以挂多个 tag——从多个主题入口都能被找到。

## 2. frontmatter 规范

每篇文档**必须**有 frontmatter：

```yaml
---
date: 2026-06-04
tags:
  - vllm-ascend
  - 编译安装
---
```

博客文章（`blog/posts/`）额外加 `categories` 和可选 `authors`：

```yaml
---
date: 2026-06-25
categories:
  - 复盘
authors:
  - xuchi
tags:
  - Qwen3
  - 精度
---
```

作者列表在 `docs/blog/.authors.yml`，新增作者往里加一条即可。

## 3. tag 命名原则

- **专有名词 / 缩写 / 工具名** 用英文：`Qwen3` `DeepSeek` `MLA` `MoE` `TP` `DP` `vllm-ascend` `CANN`
- **通用概念 / 操作 / 场景** 用中文：`编译安装` `部署` `量化` `故障排查` `双机` `精度`
- **核心原则：不要语义重复**，每个概念只用一个 tag
- 中英文都常用 → 中文；英文更常用 → 英文
- 避免过泛的 tag（如「工具」「文档」），用更具体的词

### 标准 tag 词表

| 维度 | tags |
| --- | --- |
| 模型 | Qwen3, DeepSeek |
| 框架/工具 | vllm-ascend, vllm, CANN, AISBench, NetLoader, aria2c |
| 并行/技术 | MLA, MoE, TP, DP, MTP, PD分离, 并行, lm-head |
| 操作 | 编译安装, 部署, 量化, 下载, 源码追踪 |
| 场景 | 双机, 并发, 挂起, 环境 |
| 质量 | 精度, 性能 |
| 流程 | 测试, CI, nightly, 故障排查 |
| 数据集 | GSM8K |

> 词表随内容增长扩展。新增 tag 前先确认现有词表里没有同义的。

## 4. 写作模板

每类有推荐结构，照着写能降低读者理解成本。

### 操作指南

按顺序写：

1. 前置条件（环境 / 权限 / 依赖）
2. 步骤（可复制命令 + 说明）
3. 验证（怎么确认成功）
4. 常见问题（FAQ / 踩坑）

### 参考手册

- 一事一文档（一个 API / 参数 / 术语一页）
- 准确、完整、可查阅
- 不讲「为什么」（那是原理解析的事）

### 原理解析

1. 背景（为什么有这个东西）
2. 机制（怎么工作）
3. 取舍（优点 / 代价 / 对比方案）

### 设计方案

frontmatter 加 `status` 字段（取值 `draft` / `accepted` / `superseded`）。正文：

1. 问题 / 动机
2. 方案设计
3. 影响

### 博客

- 有观点、有叙事
- 加 `<!-- more -->` 标记摘要分割点（索引页只显示摘要）
- 署名走 `authors` 字段

## 5. 新增分类的门槛

如果想新增顶层分类，必须先证明**现有 5 类都装不下这个内容**。先尝试用 tag 或 frontmatter 字段解决，实在不行才加目录。这能防止目录膨胀。

## 6. 首页 News 与公告横幅维护

### News（内容更新日志）

首页（`docs/index.md`）的 News 表格**只保留最新 5 条**公告：

- 新公告插到表格最上方，超过 5 条时删掉最旧的一行
- 每条公告包含：日期（YYYY-MM-DD）+ 做了什么（有新增/修改文档时给站内相对链接）
- 有新文档发布或重要内容更新时**同步更新 News**，与内容 commit 放在同一次提交里

### 公告横幅（当前有效的全站通知）

顶部公告横幅定义在 `overrides/main.html` 的 `{% block announce %}`，两者分工：**News 记录"内容变了什么"，横幅通知"现在正在发生什么"**（服务故障、维护窗口、征稿等需要全站可见的时效性信息）。

- 优先单条；确有多条同时有效时用轮播（`overrides/main.html` 的 `.announce-slider` 里多放一个 `<p>` 即可），**最多 3 条**，每条写清「什么事 + 影响 + 预期」（何时恢复/是否需要读者做什么）
- 轮播对读者是整体一条横幅：**修改任何一条都会让整个横幅对所有读者重新弹出**，所以只在真正需要全员知晓时改动，错别字修正除外
- 读者点 ✕ 后，本次浏览器会话内不再出现（sessionStorage，非 cookie）；下次打开浏览器重新提醒一次
- 公告失效（如故障恢复）后清空 block 内容，横幅整条消失；若值得留档，在 News 表补一行
- 短暂、只与单篇内容相关的说明放该文档内，不占横幅

## 7. 提交方式

1. Fork 仓库 → 新建分支
2. 写文档：放对分类、加 frontmatter、套模板
3. 本地验证：`mkdocs build` 无 error；`markdownlint-cli` 对改动文件无报错（若用 `markdownlint-cli2` 需自行迁移禁用规则配置，否则对已禁用规则报假阳性）
4. 提交 PR

内容红线：文档**不含敏感信息**（账号、密码、内网 IP 等）。

文件命名：`docs/` 下统一**英文**、全小写 + 短横线（✅ `building-vllm-ascend-from-source.md`，❌ `编译安装指南.md` / `编译安装 v2 最终版.md`）。

commit 规范：

- 用 [conventional commits](https://www.conventionalcommits.org/)：`feat:` / `fix:` / `docs:` / `refactor:` / `chore:`
- 加 `-s`（Signed-off-by）

## 站点维护踩坑（维护者笔记）

本地开发与构建的实战经验，遇到怪现象先查这里：

- **`mkdocs serve` 改 `mkdocs.yml` 后配置不生效**：热重载不更新配置（nav/插件/favicon/logo），要停掉 serve、删 `site/` 再重启。
- **改 `overrides/`（主题覆盖）不热重载**：启动 serve 加 `--watch-theme`，或 touch `mkdocs.yml` 触发重建。
- **meta 插件必须在 blog 之前**：否则 `.meta.yml` 的默认值（如 blog/posts 默认 authors）不被 blog 插件读取。
- **blog 阅读时间关闭**：CJK 分词不支持，中文文章阅读时间不准，已设 `post_readtime: false`。
- **git mv 后未 commit 时 git-revision-date 报时间戳 warning**：staged rename + unstaged modify 的中间状态导致，commit 后消失，benign。
- **Giscus 主题 CSS 用 commit hash**：`overrides/partials/comments.html` 的主题 URL 钉在 `@1ba9b7a`。改 `docs/giscus-theme-*.css` 后必须同步换 hash，否则 jsDelivr 返回缓存旧 CSS。
- **Giscus 评论按 pathname 匹配 + RTD 版本前缀坑**：RTD 给所有页面加 `zh-cn/latest/` 前缀，本地预览无前缀，同一页在本地和 RTD 是两份 discussion。已用 `data-strict="1"` 精确匹配防止跨页串评；首页关评论（`comments: false`）避免 `/` 模糊匹配。验证线上页面时 curl 要带 `-L` 跟随重定向。
- **busuanzi 本地总访问量数字异常**：本地 `127.0.0.1` 是全球本地调试共用的域名，数字被污染；线上独立域名正常，本地数字直接忽略。
- **favicon 位图改色**：无矢量源，用 alpha 合成（Pillow + numpy）。PIL 存 `.ico` 会缩到 64x64，要 `save(format='PNG')` 保留原尺寸。
- **公告横幅关闭状态不能靠主题自带判断**：Material 对横幅整块 DOM 取 hash，轮播挪 `is-active` class 会让 hash 不稳、换页后横幅诈尸。`overrides/main.html` 里改为按各条公告**文字**取 hash，自存 `sessionStorage`（每次浏览器会话提醒一次）。改公告逻辑时别退回原生机制。

## 目录结构速查

```text
docs/
├── guides/          # 操作指南（做）
│   ├── environment/
│   ├── deployment/
│   ├── benchmarking/
│   ├── tuning/
│   └── troubleshooting/
├── reference/       # 参考手册（查）
├── explanations/    # 原理解析（懂）
├── design/          # 设计方案（议）
├── blog/posts/      # 博客（看）
├── tags.md          # 标签索引
└── about.md
```

有问题或想法，欢迎在 [Discussions](https://github.com/xuchi-0808/Ascend-Inference-wiki/discussions) 提出。
