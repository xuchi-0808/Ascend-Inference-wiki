---
comments: false
---

# Ascend Inference Wiki

昇腾（Ascend）大模型推理开发经验沉淀 —— 面向社区与 AI Agent 的 AI Infra 知识库。

- [开始阅读](guides/environment/building-vllm-ascend-from-source.md)
- [GitHub 仓库 ↗](https://github.com/xuchi-0808/Ascend-Inference-wiki)

---

## News

| 日期 | 更新 |
| --- | --- |
| 2026-11-10 | 内容清理：把《Qwen3-235B-A22B 双机 vllm serve 手动启动命令》《从 nightly CI 流水线日志提取测试拉起方式》《双机 nightly 测试手动拉起经验总结》《vllm-ascend 双机环境调试踩坑记录》和三篇早期原理解析（DSV3.2 O 矩阵切分、O-Proj Module-Level TP、lm-head TP 源码追踪）挪进了[博客](blog/index.md)，保留原写作日期，排在时间线后段；操作指南不再分子类，两篇指南直接列出；博客从此兼收旧文档存档。 |
| 2026-10-31 | 页面顶部加了公告栏：多条公告轮流显示，点 ✕ 关掉后本次浏览不再弹出；公告栏和 News 各管什么，写进了贡献指南（CONTRIBUTING）。 |
| 2026-08-24 | 新增 [《vLLM Ascend Profiling 分析指南》](explanations/ascend-profiling-analysis.md)（原理解析）：讲怎么读 profiling 产物（`kernel_details.csv`、`trace_view.json`），配了共享专家多流 CV 掩盖、host bound 空泡定位、多卡对齐与快慢卡分析三个实际案例，附 36 张配图。 |
| 2026-08-18 | 新增 [《vLLM 权重加载与基础并行策略（TP、EP）》](explanations/vllm-weight-loading-tp-ep.md)（原理解析）：讲清楚一份 checkpoint 是怎么变成每张卡上的权重切片的——loader 怎么选、EP 下怎么过滤、WeightsMapper 怎么映射、权重按 TP/EP 怎么切，以及怎么加快加载、跨实例怎么取权重，附流程图。 |
| 2026-08-03 | 整站按读者目的重新分成五大栏目——「做 / 查 / 懂 / 议 / 看」，加了标签系统和博客；同时补了[信息架构设计文档](design/wiki-information-architecture.md)和贡献指南（CONTRIBUTING）。 |

---

## 内容导航

- **操作指南** —— 跟着步骤完成任务，读者来「做」。[浏览 →](guides/index.md)
- **参考手册** —— 查阅 API、参数、规格，读者来「查」。[浏览 →](reference/index.md)
- **原理解析** —— 理解机制与设计取舍，读者来「懂」。[浏览 →](explanations/index.md)
- **设计方案** —— 方案与 RFC，读者来「议」。[浏览 →](design/index.md)
- **博客** —— 复盘、笔记、存档，读者来「看」。[浏览 →](blog/index.md)

---

## 关于本站

本站由 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 构建，托管于 [Read the Docs](https://readthedocs.org)。文档以 Markdown 编写，源码在 [GitHub](https://github.com/xuchi-0808/Ascend-Inference-wiki)。

## 评论与交流

每页底部接入 **Giscus** 评论系统（基于 GitHub Discussions）。有问题或补充，欢迎用 GitHub 账号在页面下方留言 —— 评论同步存档到仓库的 Discussions，便于追溯。
