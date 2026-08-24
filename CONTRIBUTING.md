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

## 6. 首页 News 维护

首页（`docs/index.md`）的 News 表格**只保留最新 5 条**公告：

- 新公告插到表格最上方，超过 5 条时删掉最旧的一行
- 每条公告包含：日期（YYYY-MM-DD）+ 做了什么（有新增/修改文档时给站内相对链接）
- 有新文档发布或重要内容更新时**同步更新 News**，与内容 commit 放在同一次提交里

## 7. 提交方式

1. Fork 仓库 → 新建分支
2. 写文档：放对分类、加 frontmatter、套模板
3. 本地验证：`mkdocs build` 无 error
4. 提交 PR

commit 规范：

- 用 [conventional commits](https://www.conventionalcommits.org/)：`feat:` / `fix:` / `docs:` / `refactor:` / `chore:`
- 加 `-s`（Signed-off-by）

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
