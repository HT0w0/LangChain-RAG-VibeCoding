# 百炼平台接入实测笔记

记录接入阿里云百炼（DashScope）过程中**亲手验证**的结论，以及被证伪的传闻。

> 机器可读的版本在 [`API-CHECK.md`](API-CHECK.md)，由 `backend/scripts/check_api.py`
> 自动生成。本文是人工整理的**解释**，那份是**原始数据**。两者冲突时以那份为准。

---

## 1. 端点拓扑

一个百炼**工作空间**会同时暴露多个网关，协议不同、能力集也不同。本项目实测到的：

| 用途 | 端点 | 协议 | 结果 |
|---|---|---|---|
| Chat | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容 | ✅ 200 |
| Chat | `https://ws-<id>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容 | ✅ 200 |
| Chat | `https://ws-<id>.cn-beijing.maas.aliyuncs.com/apps/anthropic` | Anthropic 兼容 | ✅ 认证通过 |
| Embedding | `.../compatible-mode/v1/embeddings` | OpenAI 兼容 | ✅ 200 |
| Embedding | `https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding` | 原生 | ✅ 200 |
| Rerank | `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank` | 原生 | ✅ 200 |
| Rerank | `.../compatible-mode/v1/rerank` | — | ❌ **404** |

**要点**

- 工作空间的 `/models` 返回与公共端点**完全一致**的 262 个模型，说明网关是同一套后端的不同协议门面。
- **重排没有兼容端点**。`/v1/rerank` 返回 404，必须走原生端点，请求体是 DashScope 自己的嵌套结构：

  ```json
  {
    "model": "qwen3.7-text-rerank",
    "input": {"query": "...", "documents": ["...", "..."]},
    "parameters": {"top_n": 5, "return_documents": false}
  }
  ```

  这意味着**重排无法复用 `langchain-openai`**，必须手写 `httpx` 客户端。

- Anthropic 兼容网关实测认证通过（返回 `Model not exist` 是模型名问题，说明请求已穿过认证层）。
  这是本项目 Provider 层保留 `AnthropicCompatLLM` 备选实现的原因——可演示
  「同一套 RAG 逻辑切换两种协议」。

---

## 2. 模型事实

| 模型 | 结论 |
|---|---|
| `qwen3.7-plus` | 可用。**是思考模型**，默认返回 `reasoning_content` |
| `qwen3.7-text-embedding` | 可用，**1024 维**，支持批量，中英文均可 |
| `text-embedding-v4` | 可用，1024 维，批量可用 |
| `qwen3.7-text-rerank` | 可用，区分度显著优于 `gte-rerank-v2` |

### 关闭思考

`qwen3.7-plus` 默认会先产出长段推理。RAG 场景不需要长链推理，且思考会显著推高
**首 token 时延**（SSE 体验里最关键的指标）。

实测两种写法都能关闭：

```json
{"enable_thinking": false}
{"thinking": {"type": "disabled"}}
```

关闭后响应中不再出现 `reasoning_content`。

### 重排模型选择：区分度比绝对值重要

同一组「1 篇相关 + 2 篇不相关」文档上的实测得分：

| 模型 | 相关文档 | 不相关文档 | 区分度 |
|---|---|---|---|
| `qwen3.7-text-rerank` | **0.978** | 0.099 / 0.062 | **0.879** |
| `gte-rerank-v2` | 0.390 | 0.025 | 0.365 |

（qwen 的这组数字来自 `check_api.py` 的固定用例；gte 那组来自更早的一次手工对比，
用例不同，只作量级参考。）

区分度直接决定**阈值截断能不能用**。gte 的相关文档得分只有 0.39，设任何阈值都会
误杀；qwen 的 0.98 让「低于某分数直接丢弃」成为可行策略——这在过滤噪声片段上
很关键，重排后拿 top5 里混进一篇不相关的，会直接污染生成。

> 另一个反直觉的点：**rerank 模型也出现在 `/models` 列表里**（262 个模型包含它），
> 所以模型清单校验不能只对 chat/embedding 做。

---

## 3. 被证伪的传闻

### ❌「OpenAI 兼容端点不支持 Embedding」

调研资料里这条流传很广。**实测为假**：

```
POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
{"model": "qwen3.7-text-embedding", "input": ["文本1", "文本2"]}
→ 200，返回 1024 维向量
```

`input` 同时接受字符串与数组，中文正常，实测一次性提交 10 条成功。

这条纠正的意义不在于省了多少事，而在于它验证了一个工作方式：**平台假设必须实测，
不能靠文档和二手结论**。同一次调研里还有若干条无法当场验证的结论，最后都靠
`pip index versions` / 直连 API 拿到了真值。

### ⚠️ 一个测试方法本身的坑（而非平台限制）

初次测试批量时，`input: ["测试文本"]` 返回 400，一度被判断为「不支持数组」。
真实原因是 **shell 在引号处理时破坏了中文字符**。改用 Python 直接发请求后批量正常。

**教训**：命令行里传中文 JSON 不可靠，验证类脚本一律用 Python 的 HTTP 客户端写，
并显式 `ensure_ascii=False` + UTF-8 编码。

---

## 4. 错误码语义

| 场景 | HTTP | `error.code` | 是否可重试 |
|---|---|---|---|
| 未提供 Key | 401 | `InvalidApiKey` / `No API-key provided.` | ❌ |
| Key 无效 | 401 | `InvalidApiKey` | ❌ |
| **欠费/额度耗尽** | **400** | `Arrearage` | ❌ |
| 限流 | 429 | `Throttling` | ✅ |
| 网关故障 | 502/503 | —（可能是 HTML 错误页） | ✅ |

> **欠费是 400 而不是 401。** 这条容易踩：按状态码判断会把它当成「请求参数写错了」，
> 然后去翻请求体，而真正的问题是余额。必须解析 `error.code`。
>
> 同理，网关故障时可能返回 **HTML 而非 JSON**，错误解析必须容错，否则真正的
> 502 会被一个 `JSONDecodeError` 掩盖。

代码中的对应实现见 `app/core/exceptions.py`（`QuotaExhaustedError`）与
`app/providers/embedding.py`（`_extract_upstream_error`）。

---

## 5. Key 的格式

本项目使用的 Key 形如：

```
sk-ws-<分段>.<分段>.<base64ish 的 DER 签名>
```

**这是该工作空间的正常格式**，不是编码错误或显示截断。早期曾因「与常见的
`sk-` + 32 位十六进制不符」而误判为无效——实际那一次失败是**该 Key 本身失效**，
与格式无关。

> 排查记录：用「无 Key」「构造的假 Key」「真 Key」三者做差分测试。前两者分别返回
> `"No API-key provided."` 和 `"Invalid API-key provided."`，真 Key 报错与假 Key
> **完全一致**——证明请求已到达认证层，问题在凭据本身而非端点或网络。
> 换一把 Key 后三个网关全部通过。

---

## 6. 批量上限

Embedding 的批量上限**未在文档中给出可信值**，只能实测。

`check_api.py` 用「指数上探 + 二分收敛」探测：先用 2× 步长快速框出「可用/不可用」
的区间，再在区间内二分。纯指数上探会浪费额度，纯二分又缺上界。

**实测结果：`qwen3.7-text-embedding` 单批上限为 20 条。**

探测轨迹：`20 ✓ → 40 ✗ → 30 ✗ → 25 ✗ → 22 ✗ → 21 ✗`，收敛到 20。

配置里取 16 而非 20，留安全余量——不同文本长度下服务端可能按 token 数而非条数
限流，卡在边界上会偶发失败。换模型后必须重跑 `check_api.py` 复核，这个值随模型变化。

---

## 7. Chroma 的元数据过滤（本地组件，非百炼）

Chroma 的 `where` 语法**看起来**像 MongoDB，但有一处关键差异，踩了才知道：

```python
# ❌ 抛 ValueError: Expected operator expression to have exactly one operator,
#    got {'$gte': 1000, '$lte': 5000}
{"price": {"$gte": 1000, "$lte": 5000}}

# ✅ 拆成两个独立条件再合并
{"$and": [{"price": {"$gte": 1000}}, {"price": {"$lte": 5000}}]}
```

**规则：同一个字段的操作符表达式只能有恰好一个操作符。** 区间查询必须靠 `$and`
拼。项目里的做法是把这段规则封进 `indexing/vectorstore.py` 的 `build_where()`，
调用方只传 `price_min=` / `price_max=`，不直接拼这个结构。

另一处容易忽略的：**collection 的向量空间必须显式指定为 cosine**。
Chroma 默认是 `l2`，而 L2 距离**没有上界**——本项目的语义缓存阈值（相似度 > 0.95）
建立在余弦相似度落在 `[-1, 1]` 这个前提上，用 L2 的话那个阈值就只是个凭感觉
拍出来的魔数。建 collection 时写死：

```python
client.get_or_create_collection(name=..., metadata={"hnsw:space": "cosine"})
```

还要注意**这个 metadata 只在 collection 创建时生效**。已存在的 collection
改它不会重建索引，需要删掉重建。
