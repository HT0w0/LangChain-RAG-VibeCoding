"""Rerank Provider 实现。

**重排必须走原生端点。** 兼容端点没有 `/rerank`（实测 404），所以这里用
`httpx` 手写，套不上 `langchain-openai`——那个包只认 OpenAI 的形状。

端点为 `{NATIVE_URL}/services/rerank/text-rerank/text-rerank`，
请求体是 DashScope 原生嵌套结构（`input.query` / `input.documents` /
`parameters.top_n`），不是 OpenAI 的扁平结构。

模型选型：`qwen3.7-text-rerank` 的区分度实测显著优于 `gte-rerank-v2`——
相关文档 0.823 vs 不相关 0.265/0.127，而 gte 只有 0.390 vs 0.025。
区分度直接决定阈值截断能不能用。
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence

import httpx

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.providers.base import RerankHit
from app.providers.embedding import _raise_for_upstream  # noqa: PLC2701


class DashScopeReranker:
    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._s = settings
        self._url = settings.native_rerank_url
        self._headers = {
            "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        }
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    @property
    def model_id(self) -> str:
        return self._s.RERANK_MODEL

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankHit]:
        if not documents:
            return []

        client = await self._get_client()
        payload = {
            "model": self._s.RERANK_MODEL,
            "input": {"query": query, "documents": list(documents)},
            "parameters": {
                "top_n": min(top_n or self._s.RERANK_TOP_N, len(documents)),
                # 不回传原文：候选正文调用方手里就有，回传只会白白撑大响应体
                "return_documents": False,
            },
        }

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await client.post(self._url, headers=self._headers, json=payload)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt == 2:
                    raise ProviderError(
                        f"连接百炼 rerank 端点失败：{exc}",
                        provider="dashscope-rerank",
                        retryable=True,
                    ) from exc
            else:
                if resp.status_code == 200:
                    return self._parse(resp)
                try:
                    _raise_for_upstream(resp, "dashscope-rerank")
                except ProviderError as exc:
                    if not exc.retryable or attempt == 2:
                        raise
                    last_exc = exc

            await asyncio.sleep((2**attempt) * 0.5 + random.uniform(0, 0.25))

        raise ProviderError(  # pragma: no cover
            f"rerank 重试耗尽：{last_exc}", provider="dashscope-rerank", retryable=True
        )

    @staticmethod
    def _parse(resp: httpx.Response) -> list[RerankHit]:
        body = resp.json()
        # 原生端点：{"output": {"results": [{"index": n, "relevance_score": f}]}}
        results = (body.get("output") or {}).get("results")
        if results is None:
            # 兜底：个别版本直接在顶层给 results
            results = body.get("results")
        if not isinstance(results, list):
            raise ProviderError(
                f"rerank 响应结构不符合预期：{list(body)[:5]}", provider="dashscope-rerank"
            )

        hits = [
            RerankHit(
                index=int(r.get("index", 0)),
                score=float(r.get("relevance_score", r.get("score", 0.0)) or 0.0),
                document=r.get("document"),
            )
            for r in results
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits


class NoopReranker:
    """不重排，按原顺序截断。

    存在的意义是让 benchmark 能干净地做「开关重排」的 A/B 对比——
    关掉重排时链路其余部分必须完全一致，否则测出来的差异归因不清。
    """

    model_id = "none"

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankHit]:
        n = min(top_n or len(documents), len(documents))
        return [RerankHit(index=i, score=1.0 - i * 0.001) for i in range(n)]


class LocalCrossEncoderReranker:
    """本地重排（可选依赖），只做 CPU 路径。

    模型选 `bge-reranker-base`（1.1 亿参数，CPU 可用）。
    **不要用 `bge-reranker-v2-m3`**——5.68 亿参数、2.27GB，纯 CPU 上单次
    200-400ms 起步，会成为整条链路最慢的一环。

    注意 `CrossEncoder` **没有** `rerank()` 方法（那是任务名不是 API），
    要用 `.predict(pairs)`。CPU 上必须 `use_fp16=False`。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, device: str = "cpu") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                '本地重排需要可选依赖：pip install -e ".[local]"', provider="local-reranker"
            ) from exc

        self._model_name = model_name
        self._model = CrossEncoder(model_name, device=device, use_fp16=False)

    @property
    def model_id(self) -> str:
        return self._model_name

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankHit]:
        if not documents:
            return []

        import anyio

        pairs = [(query, d) for d in documents]
        scores = await anyio.to_thread.run_sync(lambda: self._model.predict(pairs))

        hits = [RerankHit(index=i, score=float(s)) for i, s in enumerate(scores)]
        hits.sort(key=lambda h: h.score, reverse=True)
        n = min(top_n or len(documents), len(hits))
        return hits[:n]


class RerankUnavailableError(ProviderError):
    """重排服务不可用时的降级信号——调用方应退回 RRF 顺序而不是整个请求失败。"""

    code = "rerank_unavailable"
    http_status = 503


def build_reranker(
    settings: Settings,
) -> DashScopeReranker | LocalCrossEncoderReranker | NoopReranker:
    if not settings.RERANK_ENABLED or settings.RERANK_BACKEND == "none":
        return NoopReranker()
    if settings.RERANK_BACKEND == "local":
        return LocalCrossEncoderReranker()
    return DashScopeReranker(settings)
