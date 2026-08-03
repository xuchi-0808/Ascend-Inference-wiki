# Ascend Inference Wiki

面向社区与 AI Agent 的昇腾大模型推理知识库——内容灵活、限制小，覆盖 AI Infra、推理特性、模型知识、实践经验。

[:rocket: 在线站点](https://ascend-inference-wiki.readthedocs.io/) · [:book: 贡献指南](CONTRIBUTING.md) · [:label: 标签索引](https://ascend-inference-wiki.readthedocs.io/tags/)

## 内容结构

按**读者意图**分五大类，口诀「做、查、懂、议、看」：

| 分类 | 读者来 | 放什么 |
|---|---|---|
| 操作指南 | 做 | 跟着步骤完成任务 |
| 参考手册 | 查 | 查阅事实（API、参数、规格） |
| 原理解析 | 懂 | 理解机制与取舍 |
| 设计方案 | 议 | 方案、RFC、决策 |
| 博客 | 看 | 复盘、笔记、观点 |

技术主题（DeepSeek、双机、精度等）用 **tag** 跨分类组织——一篇文档只归一处，但可挂多个 tag，从多个主题入口都能找到。

## 贡献

欢迎贡献！先看 [CONTRIBUTING.md](CONTRIBUTING.md)：按读者意图选分类 → 写 frontmatter → 套模板 → 提 PR（conventional commits + `-s`）。

## 本地预览

```bash
pip install mkdocs-material
mkdocs serve
```

打开 <http://127.0.0.1:8000>。

## 技术栈

[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) + [Read the Docs](https://readthedocs.org)，启用 tags + blog 插件。

## 相关仓库

- [Ascend](https://github.com/xuchi-0808/Ascend) — 昇腾推理开发总仓
- [AITools_for_Ascend](https://github.com/xuchi-0808/AITools_for_Ascend) — 昇腾推理工具集
- [vllm-workspace](https://github.com/xuchi-0808/vllm-workspace) — vLLM + vllm-ascend 工作空间
