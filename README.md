# Ascend-Inference-wiki

个人工作知识库——记录昇腾推理开发部期间的重要产出文档。

## Background

专注于 **Ascend NPU 大模型推理引擎** 的开发与适配工作，主要涉及：

- **vllm-ascend**：vLLM 推理框架在昇腾平台的移植与适配
- **ATB** (Ascend Tensor Builder)：昇腾算子库相关开发
- **msmodelslim**：模型压缩 / 瘦身工具
- **性能 Benchmark**：推理性能测试与调优

## Structure

```text
docs/
├── design/           # 技术设计方案
│   ├── dsv32-dp-o-matrix-column-splitting.md
│   │   — DeepSeek V3.2 O 矩阵切分方案综合分析
│   │   （vllm-ascend SFA V1 vs MindIE-LLM DP 场景）
│   │
│   └── dsv4-mla-o-matrix-module-level-tp.md
│       — DeepSeek V4 MLA Attention O-Proj Module-Level TP 方案分析
│
├── debug/            # 调测经验总结（无编号，英文名）
│   ├── environment-troubleshooting-guide.md
│   │   — 双机环境调试踩坑记录（编译/环境变量/benchmark 等）
│   ├── qwen3-235b-dual-node-accuracy-regression.md
│   │   — Qwen3-235B 双机 GSM8K accuracy 排查记录
│   └── qwen3-235b-manual-launch.md
│       — 双机 vllm serve + aisbench 手动拉起命令速查
│
└── practice/         # 实操指南
    ├── dual-node-test-manual-launch-guide.md
    │   — 从测试框架中提取环境变量和 vllm serve 命令的手动拉起方式
    │
    ├── extracting-test-procedure-from-ci-logs.md
    │   — 如何从 CI 日志中自行提取完整的测试拉起信息
    │
    └── building-vllm-ascend-from-source.md
        — vllm-ascend 从源码编译安装指南

debug/                   # 草稿区（可以有编号、非英文名）
```

## How to Use

- 使用 Markdown 编写，保持结构清晰
- 文档内可引用外部链接或关联其他文档
- 支持全文搜索（GitHub / GitCode 内置搜索）
- 建议用 Obsidian 或其他 Markdown 编辑器浏览以获得更好体验

## Related Repos

- [Ascend](https://github.com/xuchi-0808/Ascend) — 昇腾推理开发总仓（MindIE-LLM / benchmark / msmodelslim / ATB）
- [AITools_for_Ascend](https://github.com/xuchi-0808/AITools_for_Ascend) — 昇腾推理工具集
- [vllm-workspace](https://github.com/xuchi-0808/vllm-workspace) — vLLM + vllm-ascend 工作空间

## Remote

- GitHub: `git@github.com:xuchi-0808/Ascend-Inference-wiki.git` (master)
- GitCode: 待配置
