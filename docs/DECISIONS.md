# 技术选型与被否决的方案

README 里只留了一行一句的摘要表；**这里放完整的对照与理由**。

选型理由只有在对比里才有信息量 —— 所以每一条都写清「为什么不选另一个」，
以及那个被否掉的方案代价是什么。

---

## 1. LangChain 1.x，但刻意不装 `langchain-community`

`langchain-community` 已于 0.4.2 冻结（官方 issue #674 宣布停止维护），
`langchain-classic` 只是承接 legacy 代码的过渡包。两者都不引入。

代价是 BM25、RRF 融合、缓存层要自己写 —— 合计不到 150 行，换来完全可控、
可单独测试、可埋点，且不依赖冻结包。

> **BM25 本来就必须自己写**：官方 `BM25Retriever` 默认分词是 `text.split()`，
> 中文整段会变成一个 token，检索直接失效。

## 2. RAG 编排用 LCEL，不用 `create_agent`

`create_agent` 跑在 LangGraph 上，适合「模型自主决定调什么工具」的开放场景。
本系统的流程是确定的（检索 → 重排 → 生成），用它会导致：

- 检索步成为**不可埋点计时的黑盒**
- 缓存插入点难找
- 引用溯源要绕 LangGraph 的事件流

改用 LCEL 串 `Runnable`，每步独立可测、可缓存、可计时。

## 3. LLM 走 `langchain-openai.ChatOpenAI` 指向百炼兼容端点

| 方案 | 否决原因 |
|---|---|
| `langchain-community.ChatTongyi` | 随 community 冻结，不再维护 |
| `langchain-qwq.ChatQwen` | 第三方 beta 包，默认 base_url 指向**国际站**，上游拒绝为其添加 `init_chat_model` 支持 |
| 直连 HTTP | 失去流式封装与重试，且 LCEL 集成要自己做 |

兼容端点本身就是 OpenAI 协议，用 `ChatOpenAI` 支持最好、坑最少。

## 4. Embedding / Rerank 走云端，本地方案降级为可选

| | 本地 bge | 云端 `qwen3.7-*` |
|---|---|---|
| 安装体积 | torch + sentence-transformers **≈2GB** | **0** |
| 模型下载 | 400MB，需配 HF 镜像 | **0** |
| 入库速度 | 16 核 CPU 逐批编码 | 批量 API，快一个量级 |
| 离线可用 | ✅ | ❌ |

选云端为主路径。本地实现**仍保留**为同接口的备用 Provider
（`pip install -e ".[local]"`），因此默认安装不需要 torch。

**本地 Provider 的两个实现要点**（万一启用）：

- `bge-base-zh-v1.5` 的 query **必须**手加前缀 `为这个句子生成表示以用于检索相关文章：`，
  passage **不加**。该前缀不在模型配置文件里，`sentence-transformers` 不会自动补，
  漏加会掉召回。向量须 `normalize_embeddings=True`。
- 重排用 `bge-reranker-base`，**不要用 `v2-m3`**（5.68 亿参数、2.27GB，纯 CPU
  200–400ms 起）。`CrossEncoder` **没有 `rerank()` 方法**（那是任务名），
  用 `.predict(pairs)` 或 `.rank(query, docs)`；CPU 上必须 `use_fp16=False`。

## 5. Embedding Provider 用 `httpx` 而非 `OpenAIEmbeddings`

`OpenAIEmbeddings` 默认 `check_embedding_ctx_length=True`，会用 **tiktoken** 按
OpenAI 的 token 表把文本切块再平均 —— 对 DashScope 的模型这套规则完全不对，
**中文会被静默切碎并产出被平均过的向量，不报错但结果是错的**。

与其试错哪些开关要关，不如直连一个实测验证过的请求形状。

## 6. 向量库选 Chroma，而非 FAISS / LanceDB

| 约束 | Chroma | FAISS |
|---|---|---|
| 元数据过滤 | 原生 `$in/$gt/$lte/$and`，**服务端预过滤** | 仅等值，且是**后过滤**（可能凑不满 k） |
| 持久化 | `PersistentClient(path=...)` 纯文件落盘 | `load_local` 需 `allow_dangerous_deserialization=True`（pickle 反序列化开关） |
| 单条删除 | `delete(ids=[...])` 干净 | 需重建索引 |

## 7. 多知识库用单 collection + `kb_id` 预过滤

每库一个 collection 在跨库检索时要查 N 次再合并，且 collection 数无上限。
单 collection + metadata filter 更简单，且 Chroma 原生支持服务端预过滤：

```python
{"$and": [{"kb_id": {"$in": kb_ids}}, {"category": "手机"}, {"price": {"$gte": 1000}}]}
```

正好命中电商的类目 / 价格筛选需求。

**代价说清楚**：单机演示不需要分库，但如果知识库大到需要独立扩缩容，
这个选择会成为障碍 —— 那时要迁移到多 collection 或独立索引服务。

## 8. 其余

| 选择 | 理由 |
|---|---|
| argon2 而非 md5/sha1 | 加盐 + 抗 GPU 暴力破解。**JWT 里不放 role**：放进去的话，降权要等 token 过期（最长 12h）才生效 |
| SQLite WAL | 支持「读并发 + 后台写入库」，无 Docker 约束下的最简方案 |
| `chunks` 表冗余存正文 | BM25 需全量语料重建索引，从 Chroma 反查代价高；引用溯源要能按 id 直取原文 |
| 阻塞调用统一走 `anyio.to_thread` | BM25 / Chroma 是同步阻塞，在 `async def` 里直调会卡死整个事件循环 |
| 不用 `trim_messages(token_counter=model)` | `ChatOpenAI` 对 qwen 的 `get_num_tokens_from_messages` 未实现，会抛 `NotImplementedError`；上下文预算自行估算 |
| 建表用 `create_all` 而非 Alembic | 单机演示项目引入迁移工具的成本大于收益。**代价是改列不会自动迁移**，只能删库重建 —— 所以阶段推进方式是「只新增表，不改已有列」。要上生产必须换 Alembic |

---

# 性能与可观测性的几个决定

## 四层缓存，逐层拦截

| 层 | 键 | 值 | 失效条件 | 收益 |
|---|---|---|---|---|
| Embedding | 模型名 + 维度 + 角色 + 正文哈希 | 向量 BLOB | 模型变更 | 商品文案措辞高度重复，省掉最贵的编码调用 |
| 检索 | 归一化问题 + 范围 + 过滤 + 语料代次 | chunk_ids | 语料写入 | 跳过向量 + BM25 两路检索 |
| 答案 | 改写后的问题 + 范围 + 语料代次 | 答案 + 来源 | 语料写入 | 省一次完整的 LLM 调用 |
| 语义 | 问题向量的最近邻（scope 同上） | 答案 + 来源 | 语料写入 | 换个问法也复用 —— **默认关闭** |

**失效机制是「语料代次」，不是「记得调 invalidate」。** 每次分块表发生写入，
代次 +1；所有内容相关的缓存的键里都带着当前代次，旧条目自然读不到。于是
不存在「忘记失效」这种失败模式 —— 失效不是一次动作，而是键的一部分。TTL
只负责回收空间。

> 一个具体的后果：文档被删掉之后，缓存里那条答案的来源编号会指向一个不存在的
> 片段，而它照常显示成一张可展开的来源卡片，且完全不报错。

**语义缓存是四层里唯一会给出错误答案的一层，所以它默认关闭。** 前两层的键是
精确匹配，错了只是没命中；这一层是「差不多就算命中」，一个没测过的阈值会把
另一个问题的答案发出去 —— 措辞、来源编号、`grounded` 全都齐整。

阈值是实测出来的（零错配区间 `(0.7924, 0.8491]`），取 0.95 落在了区间上方，
代价是那组近义改写里只命中 5/8。全部分析与重跑方式见
[`SEMANTIC-CACHE.md`](SEMANTIC-CACHE.md)。命中时以 `answer_cache_hit` 上报。

## 可观测性：`request_metrics` 明细 + `GET /api/admin/metrics`

每次问答写一行明细（`trace_id` / 用户 / 会话 / 终态 / 总耗时 / 首 token 耗时 /
各阶段耗时 / 缓存命中 / 是否 grounded / token 消耗 / **prompt 缓存读数**），
面板接口按窗口聚合出 `p50`、`p95`、`max`、`avg`、各阶段平均值与 `cache_read_rate`。

五个刻意的决定：

| 决定 | 理由 |
|---|---|
| **失败也记，且带耗时** | 只记成功请求的分布，会把最慢的那部分（慢到出错的）整个从分母里删掉 —— p95 越优化越好看，而用户感受不到。检索期失败发生在流开始之前，最容易漏，单独接了一条 |
| `outcome` 分组计数，不只是一个错误数 | 将来加超时 / 取消这类终态时，面板不用改就能看见它 |
| p50 / p95 用**最近秩**，不是线性插值 | 同一个 n=10 样本，两种定义给出 5 和 5.5 —— 都「对」，所以定义必须钉死，否则「p95 = 1.2s」无法与别处对比。选最近秩是因为它总是返回真实存在的观测值 |
| 首 token 无记录时报 `null`，不是 `0` | `0` 会被读成「首 token 瞬间出来」—— 一个漂亮得离谱的数字，而事实是这一轮一个字都没流出来 |
| prompt 缓存读数同样区分 `null` / `0`，且命中率的分母只算**报过读数**的请求 | 两者含义相反：`0` 是「平台说这次没命中」，`null` 是「平台没说」。合并之后面板上会是一条平稳的 0% 曲线，看起来像缓存坏了，而事实是我们根本没在听。详见 [`PROMPT-CACHE.md`](PROMPT-CACHE.md) |

**表里不存问题正文。** 存了就等于造了一份影子聊天记录：它在会话归属校验之外，
任何能读指标的人都能读到所有人的提问。而面板要回答的问题（长短问题与延迟的
关系）一个字符数就够了。

边界行为由 `backend/tests/test_metrics.py` 逐条固定（空窗口、n=1、`None` 不做
分母、阶段耗时按「记过这个阶段的请求数」平均、报过读数的 `0` 与没报读数的
`None` 分开）。
