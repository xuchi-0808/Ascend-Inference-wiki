# Ascend-Inference-wiki

个人工作知识库——记录华为昇腾推理开发部期间的重要产出文档。

## Background

专注于 **Ascend NPU 大模型推理引擎** 的开发与适配工作，主要涉及：

- **MindIE-LLM**：华为大模型推理引擎，包括特性开发、性能优化、问题定位等
- **vllm-ascend**：vLLM 推理框架在昇腾平台的移植与适配
- **ATB** (Ascend Tensor Builder)：昇腾算子库相关开发
- **msmodelslim**：模型压缩 / 瘦身工具
- **性能 Benchmark**：推理性能测试与调优

## Structure

```text
docs/
└── design/
    ├── DSV32_DP场景O矩阵列切方案分析.md
    │   — DeepSeek V3.2 O 矩阵切分方案综合分析
    │   （vllm-ascend SFA V1 vs MindIE-LLM DP 场景）
    │
    └── DSV4_MLA_O矩阵Module-Level_TP方案分析.md
        — DeepSeek V4 MLA Attention O-Proj Module-Level TP 方案分析
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
