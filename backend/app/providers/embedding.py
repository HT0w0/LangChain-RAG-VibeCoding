"""Embedding Provider 实现。

**主路径走 `httpx` 直连兼容端点的 `/embeddings`，而不是
`langchain_openai.OpenAIEmbeddings`。** 这是一次有意的偏离，理由如下：

`OpenAIEmbeddings` 默认 `check_embedding_ctx_length=True`，会用 **tiktoken**
把文本按 OpenAI 的 token 表切块再平均——对 DashScope 的模型来说，这套切分
规则完全不对，中文会被静默切碎并产出被平均过的向量，**不报错但结果是错的**。
关掉该开关后还有一个 `encoding_format` 默认值（base64）与 DashScope 兼容性
未知。与其在「猜哪些开关要关」上试错，不如直连一个我已实测验证过的请求
形状：`POST {base_url}/embeddings`，body `{"model": ..., "input": [...]}`。

LangChain 在检索编排层仍有价值，但在这一层它带来的只有适配风险。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from collections.abc import Sequence

import httpx

from app.core.config import Settings
from app.core.exceptions import ProviderError, QuotaExhaustedError

# 可重试的 HTTP 状态：限流 + 服务端临时故障
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# 上游错误码 -> 不可重试的语义化异常
_QUOTA_CODES = {"arrearage", "insufficient_quota", "quota_exhausted", "allocationquotaexceeded"}


def l2_normalize(vec: Sequence[float]) -> list[float]:
    """L2 归一化。

    统一在客户端做，好处有二：
    1. 云端与本地两套实现的向量落在同一可比空间（本地 bge 走
       `normalize_embeddings=True`，云端是否归一化未在文档中承诺）
    2. 归一化后 Chroma 的 L2 距离排序等价于余弦相似度排序，
       而余弦正是语义缓存阈值（0.95）想要的那个度量
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _extract_upstream_error(resp: httpx.Response) -> tuple[str, str]:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - 上游可能返回 HTML 错误页
        return "", resp.text[:300]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("code", "")), str(err.get("message", ""))
    if isinstance(err, str):
        return "", err
    return str(body.get("code", "")), str(body.get("message", ""))


def _raise_for_upstream(resp: httpx.Response, provider: str) -> None:
    code, message = _extract_upstream_error(resp)
    retryable = resp.status_code in _RETRYABLE_STATUS
    if code.lower() in _QUOTA_CODES:
        raise QuotaExhaustedError(
            f"百炼额度不足或已欠费：{message or code}", provider=provider, upstream_code=code
        )
    raise ProviderError(
        f"{provider} 调用失败 [{resp.status_code}] {code}: {message}",
        provider=provider,
        retryable=retryable,
        upstream_code=code,
    )


class DashScopeEmbedder:
    """百炼兼容端点的批量 embedding。

    `dim` 在首次调用后由响应实测回填——配置里写的 1024 是已知值，但真值
    以服务端返回为准。维度不符时向量库会写进错误形状的数据，必须早发现。
    """

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._s = settings
        self._url = f"{settings.DASHSCOPE_BASE_URL.rstrip('/')}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        }
        self._batch_size = max(1, settings.EMBED_BATCH_SIZE)
        self._client = client
        self._owns_client = client is None
        self._dim = settings.EMBED_DIM

    # ---------- 生命周期 ----------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # ---------- 元信息 ----------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._s.EMBED_MODEL

    # ---------- 核心 ----------

    async def _post_batch(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        """单批请求，带指数退避重试。"""
        client = await self._get_client()
        payload = {"model": self._s.EMBED_MODEL, "input": list(texts)}
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                resp = await client.post(self._url, headers=self._headers, json=payload)
            except httpx.RequestError as exc:  # 网络层失败，值得重试
                last_exc = exc
                if attempt == 2:
                    raise ProviderError(
                        f"连接百炼 embedding 端点失败：{exc}",
                        provider="dashscope-embedding",
                        retryable=True,
                    ) from exc
            else:
                if resp.status_code == 200:
                    return self._parse(resp, expected=len(texts), is_query=is_query)
                try:
                    _raise_for_upstream(resp, "dashscope-embedding")
                except ProviderError as exc:
                    if not exc.retryable or attempt == 2:
                        raise
                    last_exc = exc

            # 抖动退避，避免多客户端同时撞上限流窗口
            await asyncio.sleep((2**attempt) * 0.5 + random.uniform(0, 0.25))

        raise ProviderError(  # pragma: no cover - 循环内已 raise
            f"embedding 重试耗尽：{last_exc}", provider="dashscope-embedding", retryable=True
        )

    def _parse(self, resp: httpx.Response, *, expected: int, is_query: bool) -> list[list[float]]:
        body = resp.json()
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise ProviderError(
                f"embedding 返回条数不符：期望 {expected}，实际 "
                f"{len(data) if isinstance(data, list) else 'N/A'}",
                provider="dashscope-embedding",
            )

        # 服务端**不保证** data 顺序与输入一致，必须按 index 还原，
        # 否则向量与文本错位——这种 bug 不报错，只会让检索悄悄变差。
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        vectors = [list(map(float, d["embedding"])) for d in ordered]

        if vectors:
            actual_dim = len(vectors[0])
            if actual_dim != self._dim:
                if self._dim == self._s.EMBED_DIM:
                    # 首次实测与配置不符：以服务端为准并告警
                    import warnings

                    warnings.warn(
                        f"embedding 实测维度 {actual_dim} 与配置 EMBED_DIM="
                        f"{self._dim} 不一致，已按实测值更新。请同步 .env 与向量库。",
                        stacklevel=2,
                    )
                self._dim = actual_dim

        return [l2_normalize(v) for v in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        # 分批：实测单批 10 条可用，上限由 check_api.py 二分探测后回填配置
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            out.extend(await self._post_batch(batch, is_query=False))
        return out

    async def embed_query(self, text: str) -> list[float]:
        (vec,) = await self._post_batch([text], is_query=True)
        return vec


class LocalBGEEmbedder:
    """本地 bge 实现（可选依赖）。

    需要 `pip install -e ".[local]"`。延迟导入，未装 torch 时 import
    本模块不会炸。

    **只做 CPU 路径**（已确认决策）。本机是 AMD RX 9070 GRE，CUDA 不可用，
    ROCm 在 Windows 上不被 PyTorch 支持。DirectML 是一条可行但 ROI 不划算的
    路——见 README「已知优化点」。启用本 Provider 时要留意内存：机器只有 15.4GB。
    """

    # bge-zh 系列的 query 侧指令前缀。**不在模型配置文件里**，
    # sentence-transformers 不会自动补，漏加会明显掉召回。
    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5", *, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise ProviderError(
                '本地 embedding 需要可选依赖：pip install -e ".[local]"',
                provider="local-bge",
            ) from exc

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_name

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # 编码是 CPU 阻塞操作，必须丢线程池，否则卡死事件循环
        import anyio

        raw = await anyio.to_thread.run_sync(
            lambda: self._model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            )
        )
        return [list(map(float, v)) for v in raw]

    async def embed_query(self, text: str) -> list[float]:
        (vec,) = await self.embed_documents([self.QUERY_PREFIX + text])
        return vec


class MockEmbedder:
    """确定性本地向量化,不联网。`EMBED_BACKEND=mock` 时启用。

    **为什么值得单独写一个,而不是测试里 monkeypatch 个假函数**

    向量是检索链路的输入。如果测试里的假向量是「随机数」或「全零」,
    那么向量检索、RRF 融合、重排这几步在测试中实际上**没有被验证**——
    它们能跑通,但任何输入都跑得通,等于没测。

    这里用的是 **hashing trick + 字符 n-gram**:把文本切成字符二元组,
    哈希到固定维度的桶里,再 L2 归一化。结果是「字面越接近的文本,
    余弦相似度越高」——这正是检索测试需要的那条性质。虽然没有语义泛化
    (同义词不相似),但对「中文分词有没有生效」「BM25 与向量两路融合得对不对」
    「删掉文档后还检索得到吗」这些问题完全够用。

    另外这也让项目在**没有 API Key、没有网络**时能完整跑起来——
    对作品集来说,克隆下来就能 `pytest` 是硬需求。
    """

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim
        self.model_id = "mock-embedding"

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        # 字符二元组而不是单词:中文没有空格,二元组对中文的覆盖更均匀,
        # 而且不必依赖 jieba(分词器有自己的预热开销)。
        grams = [text[i : i + 2] for i in range(max(1, len(text) - 1))]
        for gram in grams or [text]:
            # 用 blake2b 而不是内置 hash():内置 hash 对 str 加了随机盐,
            # 每次进程启动结果都不同,向量会跨进程不稳定——缓存和断点续传
            # 都会因此失效。
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % self._dim] += 1.0
        return l2_normalize(vec)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def build_embedder(
    settings: Settings,
) -> DashScopeEmbedder | LocalBGEEmbedder | MockEmbedder:
    if settings.EMBED_BACKEND == "mock":
        return MockEmbedder(dim=settings.EMBED_DIM)
    if settings.EMBED_BACKEND == "local":
        return LocalBGEEmbedder()
    return DashScopeEmbedder(settings)
