# 百炼平台能力实测报告

> 由 `backend/scripts/check_api.py` 于 2026-10-09 12:38:03 自动生成。
> 手工编辑会被下次运行覆盖。

本文件是全项目的**事实来源**:任何平台假设在使用前都应以此处输出为准。

## 环境

| 项 | 值 |
|---|---|
| 兼容端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 原生端点 | `https://dashscope.aliyuncs.com/api/v1` |
| Chat 模型 | `qwen3.7-plus` (enable_thinking=False) |
| Embedding 模型 | `qwen3.7-text-embedding` |
| Rerank 模型 | `qwen3.7-text-rerank` |

## 检查结果

| 检查项 | 结果 | 说明 |
|---|---|---|
| Chat(流式) | ✅ | TTFT 871 ms |
| Embedding(批量) | ✅ | 1024维 · 批16 OK |
| Rerank | ✅ | 区分度 0.879 |
| 模型清单 | ✅ | 262 个可用模型 |

## 实测详情

### Chat(流式)

| 项 | 值 |
|---|---|
| 非流式延迟 | `1254 ms` |
| usage | `{'total_tokens': 54, 'completion_tokens': 32, 'prompt_tokens': 22, 'prompt_tokens_details': {'cached_tokens': 0, 'text_tokens': 22}}` |
| 思考字段 | `无` |
| 首token时延 | `871 ms` |
| 流式块数 | `11` |
| 流式见到思考内容 | `False` |
| 输出样例 | `RAG（检索增强生成）是一种通过先检索外部知识库中的相关信息，再将其作为上下文输入给大语言模型以生成更准确回答的技术架构` |
| reasoning_content 残留 | `False` |

### Embedding(批量)

| 项 | 值 |
|---|---|
| 维度 | `1024` |
| 单条延迟 | `157 ms` |
| usage | `{'prompt_tokens': 25, 'total_tokens': 25}` |
| 批量 | `16 条 / 687 ms` |
| 返回顺序可信 | `True` |
| 批量上限探测 | `已跳过(--quick)` |

### Rerank

| 项 | 值 |
|---|---|
| 延迟 | `146 ms` |
| 相关文档得分 | `0.9782` |
| 不相关文档得分 | `[0.099, 0.0624]` |
| 区分度(相关-最高不相关) | `0.8793` |
| 排序正确 | `True` |

### 模型清单

| 项 | 值 |
|---|---|
| 可用模型数 | `262` |
| 配置的模型缺失 | `无` |
