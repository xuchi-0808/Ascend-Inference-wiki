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
| 2026-08-24 | 新增 [《vLLM Ascend Profiling 分析指南》](explanations/ascend-profiling-analysis.md)（原理解析）：profiling 产物结构、`kernel_details.csv` 与 `trace_view.json` 解读方法，以及共享专家多流 CV 掩盖、host bound 空泡定位、多卡对齐与快慢卡分析三个典型案例，36 张配图。 |
| 2026-08-18 | 新增 [《vLLM 权重加载与基础并行策略（TP、EP）》](explanations/vllm-weight-loading-tp-ep.md)（原理解析）：从磁盘 checkpoint 到每卡参数切片的完整链路——loader 选择、惰性读盘与 EP 过滤、WeightsMapper 映射、TP/EP 切分几何，以及加载提速与跨实例取权重手段，配流程图解。 |
| 2026-08-03 | 站点信息架构重构：按读者意图分为「做 / 查 / 懂 / 议 / 看」五大分类，引入 tag 体系与博客插件，重命名导航分类；新增[信息架构设计文档](design/wiki-information-architecture.md)与贡献指南（CONTRIBUTING）。 |
| 2026-07-30 | 新增 [aria2c 从 CMC 多线程下载权重指南](guides/environment/download-from-cmc-with-aria2c.md)（操作指南）。 |
| 2026-07-29 | 站点改版上线：MkDocs Material 主题、Giscus 页面评论、阅读量统计与暗色模式。 |

---

## 内容导航

- **操作指南** —— 跟着步骤完成任务，读者来「做」。[浏览 →](guides/index.md)
- **参考手册** —— 查阅 API、参数、规格，读者来「查」。[浏览 →](reference/index.md)
- **原理解析** —— 理解机制与设计取舍，读者来「懂」。[浏览 →](explanations/index.md)
- **设计方案** —— 方案与 RFC，读者来「议」。[浏览 →](design/index.md)
- **博客** —— 复盘、笔记、观点，读者来「看」。[浏览 →](blog/index.md)

---

## 关于本站

本站由 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 构建，托管于 [Read the Docs](https://readthedocs.org)。文档以 Markdown 编写，源码在 [GitHub](https://github.com/xuchi-0808/Ascend-Inference-wiki)。

## 评论与交流

每页底部接入 **Giscus** 评论系统（基于 GitHub Discussions）。有问题或补充，欢迎用 GitHub 账号在页面下方留言 —— 评论同步存档到仓库的 Discussions，便于追溯。
