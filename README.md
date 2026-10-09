# 电商知识库 RAG 问答系统

基于 **LangChain** 的企业级检索增强生成（RAG）系统。管理员在浏览器里维护商品知识库
（说明书 / 参数表 / 售后政策），用户用自然语言提问，**答案逐句可溯源到具体资料片段**。

> **为什么做这个**：通用大模型不知道你的商品参数和价格，被问到就会编。RAG 把
> 回答约束在企业自有资料上，并给出引用——这两点决定了它在电商客服场景能不能用。

---

## 目录

- [核心特性](#核心特性)
- [架构](#架构)
- [技术选型与被否决的方案](#技术选型与被否决的方案)
- [RAG 检索链路](#rag-检索链路)
- [性能优化](#性能优化)
- [平台实测](#平台实测)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [已知优化点](#已知优化点)

---

## 核心特性

| # | 特性 | 说明 |
|---|---|---|
| 1 | 知识库管理 | 浏览器端上传/删除/重建索引，支持 PDF、Word、Excel、CSV、Markdown、TXT、HTML |
| 2 | 引用溯源 | 回答标注 `[1][2]`，侧栏展示来源卡片，可展开原文定位到具体分块 |
| 3 | 多用户多会话 | 每用户独立会话空间，服务层强制校验归属 |
| 4 | 历史持久化 | 跨登录找回历史对话，按最近活跃排序 |
| 5 | 认证与权限 | 注册/登录/改密；argon2 加盐哈希 + JWT；`admin` 独占知识库管理 |
| 6 | 企业级优化 | 三层缓存、双路并发检索、分阶段埋点（详见[性能优化](#性能优化)） |
| 7 | 反馈与指标 | 回答点赞/点踩、请求指标面板 |

---

## 架构

```
浏览器  React + TS + Vite + Ant Design
   │  Bearer JWT  ／  SSE (fetch + ReadableStream)
   ▼
FastAPI (async)
   ├─ 中间件     trace_id 注入 · 耗时埋点 · CORS · 用户级限流
   ├─ 依赖注入   JWT 解析 → current_user → require_admin
   ├─ 路由       auth │ sessions │ chat(SSE) │ kb(admin) │ admin
   ▼
Service 层
   ├─ RAGService       改写 → 混合检索 → RRF → 重排 → 组装 → 流式生成
   ├─ IngestService    解析 → 清洗 → 分块 → 向量化 → 索引
   ├─ AuthService      argon2 + JWT
   └─ SessionService   会话持久化 + 上下文窗口管理
   ▼
可插拔组件层（Protocol 接口 + 多实现）
   LLMProvider   DashScopeLLM(qwen3.7-plus) │ MockLLM
   Embedder      DashScopeEmbed(默认)       │ LocalBGE（可选）
   Reranker      DashScopeRerank(默认)      │ LocalCrossEncoder（可选）│ Noop
   ▼
SQLite (WAL) + Chroma (PersistentClient) + data/uploads/
```

**组件层为什么要做接口**：让整条链路在云端 API 不可用、额度耗尽时仍能开发与测试
（`MockLLM`）；让云端/方案可切换；让换 embedding 模型不必动检索逻辑。

---

## 技术选型与被否决的方案

每个选择都记了「为什么不选另一个」——选型理由只有在对比里才有信息量。

### LangChain 1.x，但**刻意不装** `langchain-community`

`langchain-community` 已于 0.4.2 冻结（官方 issue #674 宣布停止维护），
`langchain-classic` 只是承接 legacy 代码的过渡包。两者都不引入。

代价是 BM25、RRF 融合、缓存层要自己写——合计不到 150 行，换来完全可控、
可单独测试、可埋点，且不依赖冻结包。

> BM25 本来就必须自己写：官方 `BM25Retriever` 默认分词是 `text.split()`，
> 中文整段会变成一个 token，检索直接失效。

### RAG 编排用 LCEL，不用 `create_agent`

`create_agent` 跑在 LangGraph 上，适合「模型自主决定调什么工具」的开放场景。
本系统的流程是确定的（检索 → 重排 → 生成），用它会导致检索步成为**不可埋点计时
的黑盒**，缓存插入点难找，引用溯源要绕 LangGraph 事件流。

改用 LCEL 串 `Runnable`，每步独立可测、可缓存、可计时。

### LLM 走 `langchain-openai.ChatOpenAI` 指向百炼兼容端点

| 方案 | 否决原因 |
|---|---|
| `langchain-community.ChatTongyi` | 随 community 冻结，不再维护 |
| `langchain-qwq.ChatQwen` | 第三方 beta 包，默认 base_url 指向**国际站**，上游拒绝为其添加 `init_chat_model` 支持 |
| 直连 HTTP | 失去流式封装与重试，且 LCEL 集成要自己做 |

兼容端点本身就是 OpenAI 协议，用 `ChatOpenAI` 支持最好、坑最少。

### Embedding / Rerank 走云端，本地方案降级为可选

| | 本地 bge | 云端 `qwen3.7-*` |
|---|---|---|
| 安装体积 | torch + sentence-transformers **≈2GB** | **0** |
| 模型下载 | 400MB，需配 HF 镜像 | **0** |
| 入库速度 | 16 核 CPU 逐批编码 | 批量 API，快一个量级 |
| 离线可用 | ✅ | ❌ |

选云端为主路径。本地实现**仍保留**为同接口的备用 Provider
（`pip install -e ".[local]"`），因此默认安装不需要 torch。

### Embedding Provider 用 `httpx` 而非 `OpenAIEmbeddings`

`OpenAIEmbeddings` 默认 `check_embedding_ctx_length=True`，会用 **tiktoken** 按
OpenAI 的 token 表把文本切块再平均——对 DashScope 的模型这套规则完全不对，
**中文会被静默切碎并产出被平均过的向量，不报错但结果是错的**。与其试错哪些开关
要关，不如直连一个实测验证过的请求形状。

### 向量库选 Chroma，而非 FAISS / LanceDB

| 约束 | Chroma | FAISS |
|---|---|---|
| 元数据过滤 | 原生 `$in/$gt/$lte/$and`，**服务端预过滤** | 仅等值，且是**后过滤**（可能凑不满 k） |
| 持久化 | `PersistentClient(path=...)` 纯文件落盘 | `load_local` 需 `allow_dangerous_deserialization=True` |
| 单条删除 | `delete(ids=[...])` 干净 | 需重建索引 |

### 多知识库用单 collection + `kb_id` 预过滤

每库一个 collection 在跨库检索时要查 N 次再合并。单 collection + metadata filter
更简单，且 Chroma 原生支持服务端预过滤：

```python
{"$and": [{"kb_id": {"$in": kb_ids}}, {"category": "手机"}, {"price": {"$gte": 1000}}]}
```

### 其余

| 选择 | 理由 |
|---|---|
| argon2 而非 md5/sha1 | 加盐 + 抗 GPU 暴力破解 |
| SQLite WAL | 支持「读并发 + 后台写入库」，无 Docker 约束下的最简方案 |
| `chunks` 表冗余存正文 | BM25 需全量语料重建索引，从 Chroma 反查代价高；引用溯源要能按 id 直取原文 |
| 阻塞调用统一走 `anyio.to_thread` | BM25/Chroma 是同步阻塞，在 `async def` 里直调会卡死整个事件循环 |
| 不用 `trim_messages(token_counter=model)` | `ChatOpenAI` 对 qwen 的 `get_num_tokens_from_messages` 未实现，会抛 `NotImplementedError`；上下文预算自行估算 |

---

## RAG 检索链路

```
用户提问
 ├─ 1. 多轮改写   有历史时用 LLM 把「它的价格呢」补全为独立问题
 │                （否则检索必然跑偏）；结果缓存，避免每轮多一次调用
 ├─ 2. 并行检索   asyncio.gather(
 │                  向量路: 编码 → Chroma top20（带 kb_id/类目/价格 预过滤）
 │                  关键词路: jieba 分词 → BM25 top20
 │                )
 ├─ 3. RRF 融合   k=60，各取 top20 → 融合取 top20
 ├─ 4. 重排       qwen3.7-text-rerank → top5 → 丢掉低于 0.5 的（实测值）
 ├─ 5. 组装       按 token 预算截断、去重、附来源元数据
 ├─ 6. 生成       流式 LLM；系统提示强制「只依据资料作答 + [1][2] 标注
 │                + 资料没有就明说」
 └─ 7. 落库       助手消息 + citations + 各阶段耗时
```

**RRF** 公式 `score(d) = Σ wᵢ / (k + rankᵢ(d))`，k=60 出自
Cormack et al., SIGIR 2009。自研而非取现成实现，还因为需要保留每一路的
rank 明细用于检索质量分析。

**SSE 事件协议**——**先发 `sources` 再流 token**，用户提问后立刻看到引用了哪些
片段，而不是等答案生成完：

```
event: sources   data: {sources:[{index, chunk_id, doc_name, kb_name, score,
                                  rerank_score, snippet, category, price}]}
event: token     data: {text}
event: done      data: {message_id, answer, cited, grounded, latency_ms,
                        usage:{prompt_tokens, completion_tokens, cached_tokens},
                        cache_hit, answer_cache_hit, search_query,
                        stage_timings:{rewrite, retrieve, rerank, generate}}
event: error     data: {code, message, elapsed_ms}
```

---

## 性能优化

### 四层缓存，逐层拦截

| 层 | 键 | 值 | 失效条件 | 收益 |
|---|---|---|---|---|
| Embedding | 模型名 + 维度 + 角色 + 正文哈希 | 向量 BLOB | 模型变更 | 商品文案措辞高度重复，省掉最贵的编码调用 |
| 检索 | 归一化问题 + 范围 + 过滤 + 语料代次 | chunk_ids | 语料写入 | 跳过向量 + BM25 两路检索 |
| 答案 | 改写后的问题 + 范围 + 语料代次 | 答案 + 来源 | 语料写入 | 省一次完整的 LLM 调用 |
| 语义 | 问题向量的最近邻（scope 同上） | 答案 + 来源 | 语料写入 | 换个问法也复用——**默认关闭** |

**失效机制是「语料代次」，不是「记得调 invalidate」。** 每次分块表发生写入，
代次 +1；所有内容相关的缓存的键里都带着当前代次，旧条目自然读不到。于是
不存在「忘记失效」这种失败模式——失效不是一次动作，而是键的一部分。TTL
只负责回收空间。文档被删掉之后，缓存里那条答案的来源编号会指向一个不存在的
片段，而它照常显示成一张可展开的来源卡片，且完全不报错。

> **语义缓存是四层里唯一会给出错误答案的一层，所以它默认关闭。**
> 前两层的键是精确匹配，错了只是没命中；这一层是「差不多就算命中」，一个
> 没测过的阈值会把另一个问题的答案发出去——措辞、来源编号、`grounded`
> 全都齐整。阈值是实测出来的（零错配区间 `(0.7924, 0.8491]`），取 0.95
> 落在了区间上方，代价是那组近义改写里只命中 5/8。全部分析与重跑方式见
> [`docs/SEMANTIC-CACHE.md`](docs/SEMANTIC-CACHE.md)。命中时以
> `answer_cache_hit` 上报，保留可观测性。

### 其他

- 向量路 / 关键词路 **并发**（`asyncio.gather`）
- 阻塞调用走线程池（`anyio.to_thread.run_sync`）
- SQLite **WAL** + 关键索引 + 分页
- embedding **批量调用**，上限由 `check_api.py` 实测后回填配置

### 可观测性：`request_metrics` 明细 + `GET /api/admin/metrics`

每次问答写一行明细（`trace_id` / 用户 / 会话 / 终态 / 总耗时 / 首 token 耗时 /
各阶段耗时 / 缓存命中 / 是否 grounded / token 消耗 / **prompt 缓存读数**），
面板接口按窗口聚合出 `p50`、`p95`、`max`、`avg`、各阶段平均值与
`cache_read_rate`。五个刻意的决定：

| 决定 | 理由 |
|---|---|
| **失败也记，且带耗时** | 只记成功请求的分布，会把最慢的那部分（慢到出错的）整个从分母里删掉——p95 越优化越好看，而用户感受不到。检索期失败发生在流开始之前，最容易漏，单独接了一条 |
| `outcome` 分组计数，不只是一个错误数 | 将来加超时/取消这类终态时，面板不用改就能看见它 |
| p50/p95 用**最近秩**，不是线性插值 | 同一个 n=10 样本，两种定义给出 5 和 5.5——都「对」，所以定义必须钉死，否则「p95 = 1.2s」无法与别处对比。选最近秩是因为它总是返回真实存在的观测值 |
| 首 token 无记录时报 `null`，不是 `0` | `0` 会被读成「首 token 瞬间出来」——一个漂亮得离谱的数字，而事实是这一轮一个字都没流出来 |
| prompt 缓存读数同样区分 `null` / `0`，且命中率的分母只算**报过读数**的请求 | 两者含义相反：`0` 是「平台说这次没命中」，`null` 是「平台没说」。合并之后面板上会是一条平稳的 0% 曲线，看起来像缓存坏了，而事实是我们根本没在听。详见 [`docs/PROMPT-CACHE.md`](docs/PROMPT-CACHE.md) |

**表里不存问题正文。** 存了就等于造了一份影子聊天记录：它在会话归属校验之外，
任何能读指标的人都能读到所有人的提问。而面板要回答的问题（长短问题与延迟的
关系）一个字符数就够了。

分位数与聚合的边界行为由 `backend/tests/test_metrics.py` 逐条固定（空窗口、
n=1、`None` 不做分母、阶段耗时按「记过这个阶段的请求数」平均、
报过读数的 `0` 与没报读数的 `None` 分开）。

---

## 平台实测

**所有平台假设以 [`docs/API-CHECK.md`](docs/API-CHECK.md) 为准**——该文件由
`backend/scripts/check_api.py` 自动生成，任何结论变更都会体现在那里。

为什么值得单独强调：项目前期调研中，「OpenAI 兼容端点不支持 Embedding」这条
流传甚广的结论**被实测证伪**——`/v1/embeddings` 完全正常。文档会说错、博客会
过时、二手结论会互相抄错。所以每个假设在投入使用前都先跑一遍自检。

```bash
python backend/scripts/check_api.py
```

它会实测并输出：Chat 的 TTFT 与「关思考」是否真的生效、Embedding 的真实维度与
**批量上限**、Rerank 的**区分度**（相关 vs 不相关文档的得分差）、实际可用模型数。

### 性能基准

本地确定性部分的数字（分词冷启动、BM25 关键词路、三层缓存读写）以
[`docs/BENCHMARK.md`](docs/BENCHMARK.md) 为准，由 `backend/scripts/benchmark.py` 生成：

```bash
python backend/scripts/benchmark.py            # 只打印
python backend/scripts/benchmark.py --write    # 同时写入 docs/BENCHMARK.md
```

它**只测本地、不联网**——云端组件的延迟随网络与额度波动，复现没有意义，那部分
归上面的 `check_api.py`。这样切分是为了让「哪些数字可复现、哪些只是某次观测」
在文档里一目了然。

**系统里两个阈值都是扫出来的，不是拍的；一个平台读数也是量出来的**
（它们联网，所以不放进上面那个脚本）：

```bash
python backend/scripts/semantic_probe.py     # 语义缓存的相似度阈值 → 零错配区间
python backend/scripts/rerank_probe.py       # 重排的相关性下限 → 可答率 / 拒答率曲线
python backend/scripts/prompt_cache_probe.py # prompt 缓存到底命中没有 → 七种前缀形状
```

后两个特别值得看，它们是同一类问题的两面——**机制在、读数在，但没人看**：

- 重排阈值：截断的机制一直在（`retrieval._apply_rerank`），缺的只是一个有依据的
  数，于是 `RERANK_SCORE_THRESHOLD` 长期是 0.0 —— 等于**那一刀从来没落下过**，
  而代码里看不出来，测试也全绿。实测后取 0.5，见
  [`docs/RERANK-THRESHOLD.md`](docs/RERANK-THRESHOLD.md)。
- prompt 缓存：平台一直在回 `prompt_tokens_details.cached_tokens`，而 `Usage`
  上根本没有承接它的字段，**这个读数被默默丢掉了**。接上之后才量出结论——
  见 [`docs/PROMPT-CACHE.md`](docs/PROMPT-CACHE.md)。

---

## 快速开始

### 前置

- Python 3.11+
- Node.js 18+
- 阿里云百炼 API Key（[控制台](https://bailian.console.aliyun.com/)）

### 最短路径

```bash
# 1. 后端依赖
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple   # 默认不装 torch
cp .env.example .env             # 填入 DASHSCOPE_API_KEY

# 2. 写入示例商品资料(建表 + 建管理员 + 7 份文档,可重复跑)
python backend/scripts/seed_data.py

# 3. 起服务(两个终端)
uvicorn app.main:app --reload --app-dir backend
cd frontend && npm install && npm run dev
```

打开 http://localhost:5173,用 `admin` / `123456` 登录,勾上「示例商品资料」,
然后问「RX9070 显卡多少钱」。

> **第 2 步不是可选的。** 没有它,打开的是一个空壳:知识库列表是空的、
> 问什么都答「资料中没有找到相关内容」——而那**看起来像系统坏了**,
> 不像「系统还没内容」,两者在界面上长得一模一样。
>
> `--reset` 会先删掉同名知识库再重建(只删这一个库,不碰账号和其他库)。

### 平台自检(可选,但建议先跑)

```bash
python backend/scripts/check_api.py   # 验证 key、模型名、embedding 批量上限
```

- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/health/ready

### 前端

```bash
cd frontend
npm run dev         # http://localhost:5173
npm run test        # vitest,离线
npm run build       # tsc + vite build
```

开发期由 Vite 把 `/api` 代理到 `127.0.0.1:8000`——**浏览器眼里前后端同源**,
于是不需要任何跨域协商,SSE 也不会踩到预检请求。后端的 CORS 中间件仍然留着,
因为生产部署时前后端常常不同源。**后端要先起来。**

> 前端**只测会静默出错的那部分**(SSE 分帧、引用编号→来源卡片的映射、
> 错误体翻译、markdown 里的引用替换),外加一条把整棵树挂起来、
> 逐页渲染的冒烟。组件样式错了肉眼一看就知道,而上面那几处错了页面照常
> 显示,只是显示得不对。见 `frontend/src/**/*.test.ts`。

### 无额度开发

`.env` 里设 `LLM_BACKEND=mock`，整条链路可离线跑通——生成步骤固定，
把检索问题暴露出来。RAG 的质量问题十有八九出在检索而非生成。

### 启用本地模型（可选）

```bash
pip install -e ".[local]"
# .env: EMBED_BACKEND=local  RERANK_BACKEND=local
```

> 本地路径**只做 CPU**。本机是 AMD RX 9070 GRE，CUDA 不可用，ROCm 在 Windows
> 上不被 PyTorch 支持。DirectML 是可行但 ROI 不划算的一条路，见[已知优化点](#已知优化点)。
> 注意内存占用：机器仅 15.4GB。

---

## 项目结构

```
backend/
  app/
    main.py                  应用装配
    core/       config · security · logging · deps · exceptions
    db/         models · session · repositories/
    api/        routers/{auth,sessions,chat,kb,admin,health}.py
    schemas/    pydantic 请求/响应模型
    services/   rag · ingest · auth · session · cache · metrics
    providers/  base(Protocol) · llm · embedding · reranker
    indexing/   chunking · bm25 · vectorstore · fusion
  tests/
  scripts/      seed_data.py(示例数据) · check_api.py · smoke_e2e.py · benchmark.py
                semantic_probe.py · rerank_probe.py · prompt_cache_probe.py
frontend/
  src/
    api/          client · endpoints · sse · citations · errors · chat · types
    components/   AnswerText(引用替换) · SourceCards · AppLayout · RequireAuth
    pages/        Login · Register · Chat · KnowledgeBases · Users · Metrics · Profile
    store/        auth.tsx(登录状态 + token 还原)
data/           app.db · chroma/ · uploads/   (gitignored)
docs/           API-CHECK.md · BENCHMARK.md · PLATFORM-NOTES.md
                SEMANTIC-CACHE.md · RERANK-THRESHOLD.md · PROMPT-CACHE.md
```

---

## 已知优化点

按 ROI 主动不做的事，和做了的事一样是工程判断。

| 优化点 | 为什么现在不做 |
|---|---|
| DirectML 加速本地 embedding | 需额外引入 `onnxruntime-directml` + 模型 ONNX 导出流程，且该包版本滞后主包 6 个版本（1.24.4 vs 1.30.0）。对一条**备用**路径投入产出比不成立 |
| ONNX / OpenVINO 量化 | 还没测出瓶颈就做量化属于过早优化。`benchmark.py` 会先给出各阶段耗时分布 |
| 多路召回（如 HyDE、多查询） | 当前双路 + 重排已够用，加了会让链路复杂度翻倍而收益未经验证 |
| `request_metrics` 的定时清理 | `purge_older_than()` 已实现且有测试，但**没接定时任务**。单机演示的量级下不构成问题；加一个不跑的后台任务反而会让人误以为已经收口 |
| 重排阈值的规模化重扫 | 0.5 扫自 22 段语料 / 28 个问题。候选越多，「某条不相关的恰好分很高」的概率越大，而下界正是由那条决定的——**真实知识库规模下这个数只会偏高**。当前量级下它两次运行逐位可复现，所以先按测得的值用；重扫的入口是 `rerank_probe.py`，换重排模型也必须重跑 |
| 围绕 prompt 缓存做工程改造 | 实测在生产的形状下**看不到收益**（`docs/PROMPT-CACHE.md`），而且绝对值也不大：真实链路的 prompt 只有 286 token、其中 system 244。已经把读数接进面板——**让真实流量的 `cache_read_rate` 说话**，比拿手写语料扫出来的表去改架构可信 |
| 前端按路由拆包 | 当前单包 **1.33MB（gzip 422KB）**，几乎全是 antd。演示项目首屏快慢无所谓，而拆包要动路由结构、加 `Suspense`，收益是「有网速焦虑的人少等一秒」。真要上线，第一步是换成按需引入 + 把 `Metrics`/`Users` 这类只有管理员会进的页面动态导入 |

---

## 测试

```bash
pytest                      # 全部离线，不打真实 API
pytest -m "not slow"        # 跳过慢测试
```

单元测试全部离线：真实调用既慢、又花钱、又依赖网络，任一抖动都会让 CI 变成噪音源。
云端 Provider 的正确性验证是 `scripts/check_api.py` 的职责。

---

## License

MIT
