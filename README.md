# 电商知识库 RAG 问答系统

基于 **LangChain** 的企业级检索增强生成（RAG）系统。管理员在浏览器里维护商品知识库
（说明书 / 参数表 / 售后政策），用户用自然语言提问，**答案逐句可溯源到具体资料片段**。

> **为什么做这个**：通用大模型不知道你的商品参数和价格，被问到就会编。RAG 把回答
> 约束在企业自有资料上，并给出引用 —— 这两点决定了它在电商客服场景能不能用。

**它长什么样**：FastAPI + React 的浏览器应用。多用户、多会话、历史可找回；知识库管理
限管理员。默认**只监听本机**，整条链路可以离线跑通（见[快速开始](#快速开始)）。

---

## 快速开始

需要 **Python 3.11+** 和 **Node.js 22.12+**。API Key 可选 —— 不填也能跑（见
[无额度开发](#无额度开发)）。

> **Node 的下限是 22.12 而不是 18**,虽然很多人会默认写 18:本项目的前端工具链
> 自己声明了这个要求（`vitest@5` 要 `^22.12 || ^24 || >=26`,`vite@8` 要
> `^20.19 || >=22.12`),取严的那个。装低了会在 `npm run dev` 或
> `npm run test` 时报错,而那种报错通常看不出是 Node 版本的问题。

```bash
git clone https://github.com/HT0w0/LangChain-RAG-VibeCoding.git
cd LangChain-RAG-VibeCoding

python -m venv .venv
.venv/Scripts/activate                              # Windows
# source .venv/bin/activate                         # macOS / Linux
# `[dev]` 里是 pytest 与 ruff —— 只装 `-e .` 的话测试跑不起来
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple

cp .env.example .env                                # 填 DASHSCOPE_API_KEY(可留空)
cd frontend && npm install && cd ..
```

> 上面用的是清华镜像 —— 直连 PyPI 在国内经常卡住。
> 去掉 `-i ...` 也能用，只是慢。

**Windows 用户双击 `start.bat` 就行。** 它会检查环境、写入示例商品资料、起后端和
前端、打开浏览器，并把两个服务的日志按 `[后端]` / `[前端]` 前缀打在一个窗口里，
Ctrl+C 一起停。

```
start.bat --check          只检查环境(缺什么、怎么补,会直接印出来)
start.bat --reload         后端改代码自动重启
start.bat --no-browser     不自动开浏览器
```

不用 Windows、或想自己控制：

```bash
python backend/scripts/dev.py            # 同一个启动器,跨平台
uvicorn app.main:app --reload --app-dir backend   # 或者照旧分成两个终端
cd frontend && npm run dev
```

打开 http://localhost:5173，用 **`admin` / `123456`** 登录，勾上「示例商品资料」，
然后问「RX9070 显卡多少钱」。

> **示例数据不是可选的。** 没有它，打开的是一个空壳：知识库列表是空的、问什么都答
> 「资料中没有找到相关内容」—— 而那**看起来像系统坏了**，不像「系统还没内容」，
> 两者在界面上长得一模一样。启动器每次都跑一遍 `seed_data.py`，它自己是幂等的
> （已存在就一秒返回）；要重建加 `--reset`，那只删这一个库，不碰账号和其他库。

### 无额度开发

`.env` 里设 `LLM_BACKEND=mock`，整条链路可离线跑通 —— 生成步骤固定，把检索问题
暴露出来。**RAG 的质量问题十有八九出在检索而非生成。**

### 平台自检（可选，但建议先跑）

```bash
python backend/scripts/check_api.py   # 验证 key、模型名、embedding 批量上限
```

- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/health/ready

### 启用本地模型（可选）

```bash
pip install -e ".[local]"      # 装上 torch(约 2GB),默认安装不含
# .env: EMBED_BACKEND=local  RERANK_BACKEND=local
```

> 本地路径**只做 CPU**。本机是 AMD RX 9070 GRE，CUDA 不可用，ROCm 在 Windows 上
> 不被 PyTorch 支持。DirectML 是可行但 ROI 不划算的一条路，见[已知优化点](#已知优化点)。

---

## ⚠️ 对外之前先读这一段

`admin` / `123456` 是**演示凭据**，白纸黑字写在这份公开文档里 —— 凡是读过这个
仓库的人都知道你的管理员口令。所以**要把它放到别人能访问的地方之前，先换掉它**。

服务自己会拦**两道**，但你要知道它们各自管什么：

1. **配置**：`APP_ENV=prod` 时，发现**库里的管理员口令仍是默认值**就
   **拒绝启动**。比对的是库里的哈希而不是配置 —— 因为已存在的账号不会被
   `ensure_admin` 覆盖，**光改 `.env` 是没用的**，得删库重建或用界面上的「修改密码」。
2. **不看配置**：只要管理员口令还是默认值，**一切来自其他机器的请求
   一律 503**。这是给「绑了 `0.0.0.0` 却没想到要设 `APP_ENV`」准备的 —— 那道缝
   靠配置堵不住，只能靠「请求从哪来」这个事实。

同样的取舍也适用于 `JWT_SECRET`（默认值同样是公开的，生产环境启动即拦）。
另外默认 `CORS` 白名单只放行 Vite 的两个地址（`localhost:5173` 与 `127.0.0.1:5173`），
`/docs` 只在 `DEBUG` 时开放。

它**默认只监听 `127.0.0.1`**（`scripts/dev.py` 起 uvicorn 时没给 `--host`），
所以默认情况下别的设备根本连不上 —— 要给别人访问，你得显式绑 `0.0.0.0` 或者
部署到服务器，而那正是上面两道开始生效的时刻。

---

## 架构

```
浏览器  React 19 + TS + Vite + Ant Design
   │  Bearer JWT  ／  SSE (fetch + ReadableStream)
   ▼
FastAPI (async)
   ├─ 中间件     trace_id 注入 · 耗时埋点 · CORS · 默认口令时的远程拦截
   ├─ 依赖注入   JWT 解析 → current_user → require_admin
   │            限流:问答按用户(RATE_LIMIT_QPM) · 登录按 IP(LOGIN_RATE_LIMIT_QPM)
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
（`MockLLM`）；让云端/本地方案可切换；让换 embedding 模型不必动检索逻辑。

---

## 技术选型

每条都记了「为什么不选另一个」—— **选型理由只有在对比里才有信息量**。
完整对照与代价见 [`docs/DECISIONS.md`](docs/DECISIONS.md)。

| 决策 | 一句话理由 |
|---|---|
| LangChain 1.x，**刻意不装** `langchain-community` | 它已冻结（issue #674）、`-classic` 只是过渡包。代价是 BM25/RRF/缓存自己写，不到 150 行 |
| RAG 编排用 **LCEL**，不用 `create_agent` | 流程是确定的，用 LangGraph 会让检索步成为不可埋点的黑盒、缓存插入点难找 |
| LLM 用 `langchain-openai` 指向百炼**兼容端点** | `ChatTongyi` 随之冻结；`ChatQwen` 是 beta 且默认指向国际站 |
| Embedding / Rerank 走**云端**，本地降级为可选 | 本地要拖进 2GB 的 torch；默认安装因此不含它，换模型也不动检索逻辑 |
| Embedding 用 `httpx` 而**非** `OpenAIEmbeddings` | 后者的 tiktoken 切块对 DashScope 完全不适用 —— **中文会被静默切碎并产出被平均过的向量** |
| 向量库选 **Chroma**，而非 FAISS | 前者的元数据过滤是**服务端预过滤**，FAISS 是后过滤（可能凑不满 k），且删除要重建索引 |
| 多知识库用**单 collection + `kb_id` 预过滤** | 每库一个 collection 跨库要查 N 次；Chroma 原生支持 `$in/$and` |
| **建表用 `create_all` 而非 Alembic** | 成本大于收益。**代价是改列不会自动迁移** —— 所以只新增表、不改已有列 |

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

**RRF** 公式 `score(d) = Σ wᵢ / (k + rankᵢ(d))`，k=60 出自 Cormack et al., SIGIR 2009。
自研而非取现成实现，还因为需要保留每一路的 rank 明细用于检索质量分析。

**SSE 事件协议** —— 先发 `sources` 再流 token。**传输顺序不等于显示顺序**：前端
在答案写完之前只显示答案，来源折叠成一行「引用了 N 条资料」，点开才展开。

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

## 性能与可观测性

**四层缓存，逐层拦截**（Embedding / 检索 / 答案 / 语义，语义那层默认关闭）。

**失效机制是「语料代次」，不是「记得调 invalidate」**：每次分块表写入，代次 +1；
所有内容相关的缓存键里都带着当前代次，旧条目自然读不到。于是不存在「忘记失效」
这种失败模式 —— 失效不是一次动作，而是键的一部分。

> **语义缓存是唯一会给出错误答案的一层**（前两层是精确匹配，错了只是没命中；
> 它是「差不多就算命中」），所以默认关闭。阈值是扫出来的（零错配区间
> `(0.7924, 0.8491]`），取 0.95 落在区间上方 —— 代价是那组近义改写只命中 5/8。

**可观测性**：每次问答写一行 `request_metrics` 明细，`GET /api/admin/metrics`
按窗口聚合出 p50/p95/max/avg、各阶段均值与缓存命中率。几个刻意的决定：

- **失败也记，且带耗时** —— 只统计成功请求会把最慢的那部分（慢到出错的）整个从
  分母里删掉，p95 越优化越好看，而用户感受不到。
- **分位数用最近秩，不是线性插值** —— n=10 时两种定义给出 5 和 5.5，都「对」，
  所以定义必须钉死。选最近秩因为它总是返回真实存在的观测值。
- **`null` 和 `0` 分开**：首 token 无记录时显示 `—` 而不是 `0ms`；prompt 缓存
  读数的 `0`（平台说没命中）与 `null`（平台没说）含义相反。

完整论述见 [`docs/DECISIONS.md`](docs/DECISIONS.md)。

---

## 实测

这个项目里**每个平台假设都是跑出来验证过的**，不是从文档抄的。

| 脚本 | 它回答什么 | 结果文档 |
|---|---|---|
| `scripts/check_api.py` | 端点拓扑、真实维度、**embedding 批量上限**、重排区分度 | [API-CHECK.md](docs/API-CHECK.md) |
| `scripts/benchmark.py` | 本地确定性耗时（分词、BM25、缓存），**不联网** | [BENCHMARK.md](docs/BENCHMARK.md) |
| `scripts/loadtest.py` | **并发**：100 人同时用会怎样 | [LOAD-TEST.md](docs/LOAD-TEST.md) |
| `scripts/{semantic,rerank,prompt_cache}_probe.py` | 三个阈值/读数，联网扫出来的 | 各自一节 |


```bash
python backend/scripts/check_api.py            # 平台自检(联网)
python backend/scripts/benchmark.py            # 本地基准(不联网)
python backend/scripts/loadtest.py --selftest  # 压测器自检(19 项)
```

---

## 项目结构

```
start.bat                     一键启动(Windows,纯 ASCII——见文件头说明)
backend/
  app/
    main.py                  应用装配
    core/       config · security · logging · deps · exceptions
    db/         base(引擎/会话) · models · repositories/
    api/        routers/{auth,sessions,chat,kb,admin,health}.py
    schemas/    pydantic 请求/响应模型
    services/   rag(编排) · retrieval(检索) · ingest(摄取) · session
                auth · users · cache · metrics · ratelimit
                parsing(格式解析) · citations(引用) · context(预算)
                history(改写) · prompts(系统提示)
    providers/  base(Protocol) · llm · embedding · reranker
    indexing/   chunking · bm25 · vectorstore · fusion
  tests/                      648 条,全部离线
  scripts/      dev.py(一键启动) · seed_data.py(示例数据)
                check_api.py · smoke_e2e.py · benchmark.py
                loadtest.py(并发压测) · semantic_probe.py
                rerank_probe.py · prompt_cache_probe.py
frontend/
  src/
    api/          client · endpoints · sse · citations · errors · chat · types
    components/   AnswerText(引用替换) · SourceCards · SourceDisclosure
                  MessageBubble · SidebarList · AppLayout · RequireAuth
    pages/        Login · Register · Chat · KnowledgeBases · Users · Metrics · Profile
    store/        auth.tsx(登录状态 + token 还原)
data/           app.db · chroma/ · uploads/   (gitignored)
docs/           DECISIONS(选型与代价) · API-CHECK · BENCHMARK · LOAD-TEST
                PLATFORM-NOTES · PROMPT-CACHE · RERANK-THRESHOLD · SEMANTIC-CACHE
```

---

## 测试

```bash
pytest                                  # 后端 648 条,全部离线
cd frontend && npm run test             # 前端 64 条
```

单元测试**全部离线**：真实调用既慢、又花钱、又依赖网络，任一抖动都会让 CI 变成
噪音源。云端 Provider 的正确性验证是 `scripts/check_api.py` 的职责。

前端**只测会静默出错的那部分**（SSE 分帧、引用编号→来源卡片的映射、错误体翻译、
markdown 里的引用替换），外加一条把整棵树挂起来、逐页渲染的冒烟。组件样式错了
肉眼一看就知道，而上面那几处错了页面照常显示，只是显示得不对。

---

## 已知优化点

按 ROI 主动不做的事，和做了的事一样是工程判断。

| 优化点 | 为什么现在不做 |
|---|---|
| DirectML 加速本地 embedding | 需额外引入 `onnxruntime-directml` + ONNX 导出流程，且该包滞后主包 6 个版本。对一条**备用**路径 ROI 不成立 |
| ONNX / OpenVINO 量化 | 还没测出瓶颈就做量化属于过早优化。`benchmark.py` 会先给出各阶段耗时分布 |
| 多路召回（HyDE / 多查询） | 当前双路 + 重排已够用，加了会让链路复杂度翻倍而收益未经验证 |
| `request_metrics` 的定时清理 | `purge_older_than()` 已实现且有测试，但**没接定时任务**。加一个不跑的后台任务反而会让人误以为已经收口 |
| 重排阈值的规模化重扫 | 0.5 扫自 22 段语料 / 28 个问题。候选越多，「某条不相关的恰好分很高」的概率越大，而**下界正是由那条决定的** —— 真实规模下这个数只会偏高 |
| 围绕 prompt 缓存做工程改造 | 实测在生产的形状下看不到收益，绝对值也不大（真实 prompt 只有 286 token）。**让真实流量的 `cache_read_rate` 说话**比拿手写语料扫表可信 |
| 前端按路由拆包 | 当前单包约 1.33MB（gzip 约 420KB），几乎全是 antd。真要上线，第一步是按需引入 + 把只有管理员会进的页面动态导入 |
| SSE 端点的并发连接上限 | 限流管的是「每分钟发起多少次」，不是「同时挂多少条」。本机量级无影响，上公网前值得加一道 `CapacityLimiter` |

---

## License

MIT
